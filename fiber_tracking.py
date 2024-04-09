from math import *
import numpy as np
from scipy import ndimage, interpolate, sparse
from skimage import io, morphology, segmentation
from mayavi import mlab
import skan
import sys, os, argparse
from tqdm import tqdm
import zipfile, json
from io import BytesIO, StringIO
import argparse
#from codetiming import Timer

class System:
    def __init__(self, coords, paths, junctions, fibers, spacing):
        self.coords = coords
        self.paths = paths
        self.junctions = junctions
        self.fibers = fibers
        self.spacing = spacing
        
        # Characteristic length scale
        self.r_bar = None
        
        # Minimum length for fiber increment
        self.min_seg_length = None  
        
        #Number of voxels at junctions to discard when fitting splines
        self.n_discard_voxels = 1
        

class Path:
    def __init__(self, id, voxels, length, junc_idx1, junc_idx2):
        self.id = id
        self.voxels = voxels
        self.length = length
        self.junctions = [junc_idx1, junc_idx2]
        
        self.feature = None
        self.feature_origin = None
        self.spline = None
        self.fiber_id = None
                      
    def is_connection(self):
        return self.junctions[0] is not None and self.junctions[1] is not None
    
    def is_isolated(self):
        return self.junctions[0] is None and self.junctions[1] is None
    
    def in_fiber(self):
        return self.fiber_id is not None
    
    def __repr__(self):
        s = f'{self.__class__.__name__} id={self.id} {self.voxels}'
        if self.junctions is not None:
            s += ', junctions ' + str(self.junctions)
        return s


class DirectedPath:
    def __init__(self, path, reverse):
        self.path = path
        self.reversed = reverse
    
    @property
    def id(self): return self.path.id
    @property
    def length(self): return self.path.length
    @property
    def fiber_id(self): return self.path.fiber_id

    @property
    def junctions(self): return self.path.junctions if not self.reversed else self.path.junctions[::-1]

    @property
    def voxels(self): return self.path.voxels if not self.reversed else self.path.voxels[::-1]
    
    def is_connection(self): return self.path.is_connection()
    
    def reverse(self):
        return DirectedPath(self.path, not self.reversed)
    
    def __eq__(self, other):
        return self.path == other.path and self.reversed == other.reversed
    
    def __repr__(self):
        s = f'{self.__class__.__name__} id={self.id} {self.voxels}'
        if self.junctions is not None:
            s += ', junctions ' + str(self.junctions)
        return s


class Junction:
    def __init__(self, id, voxels, incoming_paths, outgoing_paths):
        self.id = id
        self.voxels = voxels
        self.incoming_paths = incoming_paths
        self.outgoing_paths = outgoing_paths
        
    def all_paths(self):
        return self.incoming_paths + self.outgoing_paths
    
    def __repr__(self):
        return f'Junction voxels: {self.voxels}, paths: {self.all_paths()}'
        

### Object construction ####

def find_junctions(sk, create_paths=False):
    # Create dicts containing outgoing and incoming paths for each junction voxel
    incoming_path_key = {}
    outgoing_path_key = {}
    path_objects = []
    for i in range(sk.n_paths):
        path = sk.path(i)
        outgoing_path_key.setdefault(path[0], []).append(i)
        incoming_path_key.setdefault(path[-1], []).append(i)
        if create_paths:
            path_objects.append(Path(i, path, sk.path_lengths()[i], None, None))

    junction_voxels = np.flatnonzero(sk.degrees > 2)
    
    # A subgraph containing only junction voxels
    junc_graph = sk.graph[np.ix_(junction_voxels, junction_voxels)]
    
    # Find all junctions as groups of adjacent junction voxels
    n_components, labels = sparse.csgraph.connected_components(junc_graph, directed=False)
    
    # Collect all outgoing and incoming paths for each junction
    junctions = []
    for idxs in get_components(labels):
        voxels = junction_voxels[idxs]
        outgoing_paths = sum((outgoing_path_key.get(v, []) for v in voxels), start=[])
        incoming_paths = sum((incoming_path_key.get(v, []) for v in voxels), start=[])
        i = len(junctions)
        junctions.append(Junction(i, voxels, incoming_paths, outgoing_paths))
        if create_paths:
            for idx in outgoing_paths:
                path_objects[idx].junctions[0] = i
            for idx in incoming_paths:
                path_objects[idx].junctions[1] = i

    if create_paths:
        return junctions, path_objects
    else:
        return junctions
    
def get_components(labels):
    """ Return indices of each component in a labeled sequence """
    order = np.argsort(labels)
    edges = np.flatnonzero(np.diff(labels[order])) + 1
    edges = np.r_[0, edges, len(order)]
    return [order[edges[i]:edges[i+1]] for i in range(len(edges)-1)]
    

def find_path_volumes(system, img):
    """ Use watershed to expand paths to the foreground of a binary image """
    mask = img > 0
    markers = np.zeros(mask.shape, dtype=int)
    pids = np.zeros(len(system.coords), dtype=int)
    for p in system.paths:
        pids[p.voxels[1:-1]] = p.id+1
    
    markers[tuple(system.coords.T)] = pids
    labeled = segmentation.watershed(mask, markers, mask=mask)
    regions = ndimage.find_objects(labeled)
    
    for i in range(len(regions)):
        path = system.paths[i]
        if regions[i] is not None:
            path.feature = labeled[regions[i]] == i+1
            path.feature_origin = np.array([r.start for r in regions[i]])
        else:
            path.feature_origin = np.zeros(3)

    
### Fiber recognition ####

def find_fiber(system, path):
    """ Find a fiber starting from a single path """
    seq = find_fiber_half(system, [DirectedPath(path, False)])
    seq = [p.reverse() for p in seq[::-1]]
    seq2 = find_fiber_half(system, seq)
    return seq2

def find_fiber_half(system, seq):
    """ Find one end of a fiber by continuing from a DirectedPath """
    #seq = [dpath]
    min_seg_length = system.min_seg_length
    while True:
        junc = seq[-1].junctions[1]
        if junc is None:
            break
        
        # Take a section of the end that is just longer than min_seg_length
        segs_to_include = 1
        cumlength = seq[-1].length
        while cumlength < min_seg_length and segs_to_include < len(seq):
            segs_to_include += 1
            cumlength += seq[-segs_to_include].length
         
        # Get next section
        cont = continue_fiber(system, seq[-segs_to_include:])
        if cont is None or any(cont[-1].path.id == dp.path.id for dp in seq):
            # Terminate
            break
        seq = seq + cont[1:]
    return seq


def continue_fiber(system, fib):
    """ Continue a path sequence fib by at least min_seg_length 
        (or shorter if a tail path is encountered) """
    candidates = get_continuations(system, fib)

    if len(candidates) == 0:
        return None
    
    #candidates = [filter_short_bridges(c) for c in candidates]
    
    # Calculate the energy for each potential updated version of the sequence
    energies = []
    for seq in candidates:
        try:
            energy, spline = calc_sequence_energy(system, fib + seq[1:])
        except:
            print(seq)
            raise
        energies.append(energy)
        
    # Ascending order by energy (we actually only care about the one with lowest energy)
    order = np.argsort(energies)   
    energies = [energies[i] for i in order]
    candidates = [candidates[i] for i in order]
    
    # The continuation should be optimal in both directions. Try to find another sequence 
    # in the reverse direction that has even lower energy
    end_path = candidates[0][-1].reverse()
    if end_path.junctions[1] is None:
        print(candidates[0])
    backward_candidates = get_continuations(system, [end_path])
    
    # Include only paths that do not go back to the starting point
    backward_candidates = [c for c in backward_candidates if fib[-1].id not in [p.id for p in c]]
    
    # Calculate all energies
    energies_b = []
    for seq in backward_candidates: 
        #seq = filter_short_bridges(seq)
        energy, spline = calc_sequence_energy(system, seq)
        energies_b.append(energy)
    if len(energies_b) > 0 and np.min(energies_b) < energies[0]:
        # The found continuation would not be preferred in reverse direction, so reject
        return None
    
    return candidates[0]

def filter_short_bridges(seq):
    return [p for p in seq if not (len(p.voxels) <= 4 and p.is_connection())]

def get_continuations(system, seq, max_depth=10):
    """ Find all possible continuations for seq. Continuations either end in a tail path
        or have length above min_seg_length (except if max_depth is exceeded). """
    paths, junctions = system.paths, system.junctions
    min_seg_length = system.min_seg_length
    
    # Recursive function to iteratively go deeper in depth
    def iterative_find(sequence, visited_junctions):
        last_pid = sequence[-1].id
        junc = sequence[-1].junctions[1]
        if junc in visited_junctions or len(sequence) > max_depth:
            return [sequence]
        results = []
        # All DirectedPaths emanating from the junction
        pout = junctions[junc].outgoing_paths
        pin = junctions[junc].incoming_paths
        pths = [DirectedPath(paths[p], False) for p in pout] + [DirectedPath(paths[p], True) for p in pin]
        for p in pths:
            if p.id == last_pid or p.path.fiber_id is not None or p.junctions[1] == p.junctions[0]:
                # Exclude loops and paths that are part of a fiber
                continue
            seq = sequence + [p]
            seq_length = sum(pth.length for pth in seq[1:])
            if seq_length > min_seg_length or not p.is_connection():
                # Found a termination
                results.append(seq)
            else:
                # Recursion
                results += iterative_find(seq, visited_junctions+[junc])
        return results
    
    return iterative_find(seq[-1:], [p.junctions[0] for p in seq])

def calc_sequence_energy(system, seq):
    """ Return the bending energy density of the sequence using a spline fit. If the sequence
        starts or ends with a long path, only min_seg_length of those paths is included for averaging """
    
    r_bar = system.r_bar
    min_seg_length = system.min_seg_length
    n_discard = system.n_discard_voxels
    
    def get_coords(p):
        if hasattr(p.path, 'adjusted_coords'):
            if p.reversed:
                return p.path.adjusted_coords[::-1]
            else:
                return p.path.adjusted_coords
        else:
            return system.coords[p.voxels[n_discard:-n_discard]]
    
    coords = np.concatenate([get_coords(p) for p in seq])
    if len(coords) <= 4:
        return np.inf, None
    
    tot_length = sum(p.length for p in seq)
    
    # Include only min_seg_length of the first and last paths if they're long ones
    tstart = max(seq[0].length - min_seg_length, 0)/tot_length
    tend = 1 - max(seq[-1].length - min_seg_length, 0)/tot_length
    
    # Spline fit to the entire path
    n_cpts = 2 + int(tot_length//(3*r_bar))
    
    try:
        spline = spline_fit(coords, 3, n_cpts)
    except np.linalg.LinAlgError:
        # Could not fit, probably a bad path
        return np.inf, None
    
    # Average energy from tstart to tend
    energy = spline_energy(spline, eval_range=(tstart, tend))
    return energy, spline


### Splines ####

#@Timer(name='fit', logger=None)
def spline_fit(coords, order, n_cpts):
    """ Fit a B-spline to a voxel path """
    disps = np.diff(coords, axis=0)
    dists = np.linalg.norm(disps, axis=1)
    cumdists = np.cumsum(dists)
    x_vals = np.r_[0.0, cumdists/cumdists[-1]]
    t_vals = np.r_[np.zeros(order), np.linspace(0, 1, n_cpts), np.ones(order)]
    spline = interpolate.make_lsq_spline(x_vals, coords, t_vals, k=order)
    return spline

#@Timer(name='spline energy', logger=None)
def spline_energy(c, eval_range=(0,1)):
    """ Calculate the bending energy of a spline """

    t = np.linspace(*eval_range, 1000)
    d1 = c.derivative(1)(t)
    d2 = c.derivative(2)(t)
    
    ds = np.linalg.norm(d1, axis=1)
    length = np.trapz(ds, dx=t[1]-t[0])
    energy = np.trapz(np.sum(np.cross(d1,d2)**2,axis=1) / ds**5, dx=t[1]-t[0])
    
    return energy/length


def spline_arclength(c, eval_range=(0,1)):
    """ Arc length of a spline """
    tangent = c.derivative(1)
    def func(t): return np.linalg.norm(tangent(t), axis=-1)
    t = np.linspace(*eval_range, 100)
    length = np.trapz(func(t), dx=t[1]-t[0])
    return length


### IO ####

def save_results(filename, system, compression=zipfile.ZIP_DEFLATED):
    coords, paths, junctions, fibers = system.coords, system.paths, system.junctions, system.fibers
    spacing, r_bar, min_seg_length = system.spacing, system.r_bar, system.min_seg_length
    pths = [p.voxels for p in paths]
    
    # Make path directions consistent within fibers
    #for fib in fibers:
        #for p in fib:
            #pths[p.id] = p.voxels           
    
    coords_f = BytesIO()
    paths_f = StringIO()
    junctions_f = StringIO()
    fibers_f = StringIO()
    np.savetxt(coords_f, coords, fmt='%d')
    
    for p in pths:
        paths_f.write(' '.join((str(v) for v in p)) + '\n')
    for junc in junctions:
        junctions_f.write(' '.join((str(v) for v in junc.voxels)) + '\n')
    for fib in fibers:
        fibers_f.write(' '.join((('-' if p.reversed else '')+str(p.id) for p in fib)) + '\n')

    with zipfile.ZipFile(filename, mode='w', compression=compression) as f:
        f.writestr('metadata.json', json.dumps(dict(spacing=spacing[0], r_bar=r_bar, min_seg_length=min_seg_length)))
        f.writestr('coords.txt', coords_f.getvalue())
        f.writestr('paths.txt', paths_f.getvalue())
        f.writestr('junctions.txt', junctions_f.getvalue())
        f.writestr('fibers.txt', fibers_f.getvalue())
    

def load_results(filename):
    def read_vectors(string):
        return [np.array([int(v) for v in line.split()]) for line in string.splitlines()]
    with zipfile.ZipFile(filename) as f:
        metadata = json.loads(f.read('metadata.json'))
        coords = np.genfromtxt(f.read('coords.txt').splitlines(), dtype=int)
        pths = read_vectors(f.read('paths.txt'))
        juncs = read_vectors(f.read('junctions.txt'))
        fibs = [[(abs(int(v)), v.strip().startswith(b'-')) for v in line.split()] for line in f.read('fibers.txt').splitlines()]
    
    junc_dict = {}
    for i, junc in enumerate(juncs):
        for vox in junc:
            junc_dict[vox] = i
    junctions = [Junction(i, junc, [], []) for i, junc in enumerate(juncs)]
    
    spacing = metadata['spacing']
    r_bar = metadata['r_bar']
    paths = []
    for i, p in enumerate(pths):
        disps = np.diff(coords[p], axis=0)
        dists = np.linalg.norm(disps, axis=1)
        length = np.sum(dists) * spacing
        j1 = junc_dict.get(p[0], None)
        j2 = junc_dict.get(p[-1], None)
        path = Path(i, p, length, j1, j2)
        paths.append(path)
        if j1 is not None:
            junctions[j1].outgoing_paths.append(i)
        if j2 is not None:
            junctions[j2].incoming_paths.append(i)
    
    fibers = [[DirectedPath(paths[i], reversed) for i,reversed in fib] for fib in fibs]
    
    for i,fib in enumerate(fibs):
        for pid,reversed in fib:
            paths[pid].fiber_id = i
    
    sys = System(coords, paths, junctions, fibers, np.ones(3)*spacing)
    sys.r_bar = r_bar
    return sys


### Visualization ####

def draw_sequence(system, sequence, depth, volume_opacity=0.3, annotate=True):
    """ Visualizing fibers. depth indicates how many adjancent paths are recursively drawn to visualize the neigborhood """
    paths = system.paths
    idxs = np.concatenate([traverse_network(system, p, depth) for p in sequence])
    sequence_set = set(s.id for s in sequence)
    seq_color = (0,0,0)
    for idx in np.unique(idxs):
        path = paths[idx]
        if idx in sequence_set:
            color = seq_color
        else:
            color = tuple(np.random.random(3))
        draw_path(system, path, volume_opacity=volume_opacity, color=color, annotate=annotate)

def draw_recursively(system, first_element, depth, volume_opacity=0.3, annotate=True):
    """ Visualize the neighborhood around a path or junction """
    idxs = traverse_network(system, first_element, depth)
    fiber_colors = {}
    for idx in idxs:
        path = paths[idx]
        if path.in_fiber():
            fid = path.fiber_id
            color = fiber_colors.setdefault(fid, tuple(np.random.random(3)))
        else:
            color = tuple(np.random.random(3))
        draw_path(system, path, volume_opacity=volume_opacity, color=color, draw_junctions=True, annotate=annotate)

def traverse_network(system, first_element, depth):
    paths, junctions = system.paths, system.junctions
    if type(first_element) == Junction:
        path_list = [idx for idx in first_element.all_paths()]
    else:
        path_list = [first_element.id]
    
    result = [*path_list]
    for i in range(depth):
        new_list = []
        for p_idx in path_list:
            path = paths[p_idx]
            for j_idx in path.junctions:
                if j_idx is None: continue
                junc = junctions[j_idx]
                for idx in junc.all_paths():
                    if not path in result:
                        new_list.append(idx)
        result += new_list
        path_list = new_list
    return np.unique(result)


def draw_path(system, path, volume_opacity=0.3, color=None, annotate=True):
    if color is None:
        color = tuple(np.random.random(3))
    
    # Voxel path
    w = 4
    coords = system.coords[path.voxels]
    mlab.plot3d(*coords.T, color=color, line_width=w, tube_radius=None)
    
    if hasattr(path, 'adjusted_coords'):
        mlab.plot3d(*path.adjusted_coords.T, color=color, line_width=w+4, tube_radius=None)
      
    # Volume
    if volume_opacity > 0 and path.feature is not None:
        feature = np.pad(path.feature, 1)
        ox, oy, oz = path.feature_origin - 1.5
        sx, sy, sz = feature.shape
        extent = [ox,ox+sx, oy,oy+sy, oz,oz+sz]
        mlab.contour3d(feature.astype(float), contours=[0.5], extent=extent, opacity=volume_opacity, color=color)
    
    # Junctions 
    if path.junctions[0] is not None:
        mlab.points3d(*coords[0].T, scale_factor=.3)
    if path.junctions[1] is not None:
        mlab.points3d(*coords[-1].T, scale_factor=.3)
            
    if annotate and len(coords) > 4:
        middle = coords[len(coords)//2]
        mlab.text3d(*middle, str(path.id), color=color)

def draw_spline(spline, n_pts=100, color=(0,0,0), line_width=4, draw_range=(0,1)):
    t = np.linspace(*draw_range, n_pts)
    pts = spline(t)
    mlab.plot3d(*pts.T, color=color, line_width=line_width)


def junction_diagnostics(system, pid1, pid2):
    """ A function to help diagnose a mistake made by the algorithm """
    path1 = system.paths[pid1]
    path1 = DirectedPath(path1, path1.junctions[1] is None)
    conts = get_continuations(system, [path1])
    print([[p.id for p in c] for c in conts])
    p2_conts = [c for c in conts if pid2 in [p.id for p in c]]
    
    if len(p2_conts) == 0:
        path1 = path1.reverse()
        conts = get_continuations(system, [path1])
        print([[p.id for p in c] for c in conts])
        p2_conts = [c for c in conts if pid2 in [p.id for p in c]]
    
    energies = [calc_sequence_energy(system, c)[0] for c in conts]
    imin = np.argmin(energies)
    seq = conts[imin]
    energy, spline = calc_sequence_energy(system, seq)
    
    energies = [calc_sequence_energy(system, c)[0] for c in p2_conts]
    imin = np.argmin(energies)
    energy1, spline1 = calc_sequence_energy(system, p2_conts[imin])

    path2 = [p for p in p2_conts[0] if p.id == pid2][0].reverse()
    conts = get_continuations(system, [path2])
    p1_conts = [c for c in conts if pid1 in [p.id for p in c]]
    
    energies = [calc_sequence_energy(system, c)[0] for c in p1_conts]
    imin = np.argmin(energies)
    energy2, spline2 = calc_sequence_energy(system, p1_conts[imin])
    
    print(energy, energy1, energy2)
    
    draw_spline(spline)
    draw_spline(spline1, color=(1,0,0))
    draw_spline(spline2, color=(0,1,0))
    draw_sequence(system, seq, 2)
    mlab.show()

### Script ####

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    parser.add_argument('--skeleton')
    parser.add_argument('--r_bar', type=float, default=0.0)
    parser.add_argument('--filter_stubs', action='store_true')
    parser.add_argument('output')
    args = parser.parse_args()
    imgfile = args.image
    outfile = args.output
    skelfile = args.skeleton
    metafile = imgfile + '.meta'
    img = io.imread(imgfile)
    img = img > 0
    r_bar = args.r_bar
        
    if skelfile is None:
        print('Calculating skeleton')
        skeleton = morphology.skeletonize(img)
    else:
        skeleton = io.imread(skelfile) > 0
    
    try:
        pixel_size = float(open(metafile).read().split('=')[1])
    except:
        pixel_size = 1.0
        print('Using default voxel size 1.0 µm')

    sk = skan.Skeleton(skeleton, spacing=pixel_size, source_image=img)
    
    if args.filter_stubs:
        print('Removing single voxel tails')
        for i in range(sk.n_paths):
            if len(sk.path(i)) == 2:
                for idx in sk.path(i):
                    if sk.degrees[idx] == 1:
                        skeleton[tuple(sk.coordinates[idx])] = False
        sk = skan.Skeleton(skeleton, spacing=pixel_size, source_image=img)

    total_path_length = np.sum(sk.path_lengths())
    total_volume = np.count_nonzero(img) * pixel_size**3
    if r_bar == 0:
        r_bar = np.sqrt(total_volume/total_path_length)

    min_seg_length = 4*r_bar

    print(f'r̄ = {r_bar} µm')

    junctions, paths = find_junctions(sk, create_paths=True)

    system = System(sk.coordinates, paths, junctions, [], sk.spacing)
    system.r_bar = r_bar
    system.min_seg_length = min_seg_length
    #find_path_volumes(system, img)

    fibers = system.fibers

    path_lengths = np.array([p.length for p in paths])
    length_order = np.argsort(path_lengths)[::-1]

    # Use paths longer than this as starting points for fiber tracking
    path_length_threshold = 6*r_bar
    
    n_long_paths = np.flatnonzero(path_lengths[length_order] < path_length_threshold)[0]

    for i in tqdm(length_order[:n_long_paths], 'Finding fibers'):
        path = paths[i]
        if path.fiber_id is not None or path.is_isolated():
            continue
        fib = find_fiber(system, path) 
        for dpath in fib:
            dpath.path.fiber_id = len(fibers)
        fibers.append(fib)
    
    print(f'{len(fibers)} fibers created')

    isolated_paths = [p for p in paths if p.is_isolated()]
    accepted_isolated_paths = [p for p in isolated_paths if p.length >= path_length_threshold]
    for path in accepted_isolated_paths:
        path.fiber_id = len(fibers)
        fibers.append([DirectedPath(path, False)])
    
    print(f'{len(accepted_isolated_paths)} fibers created from isolated paths')
    print(f'{len(isolated_paths)-len(accepted_isolated_paths)} isolated paths rejected for being too short')
    
    fiber_lengths = [sum(p.length for p in fib) for fib in fibers]
    print(f'{100*sum(fiber_lengths)/total_path_length:.2f}% of the skeleton assigned to fibers')
    
    save_results(outfile, system)
    print(f'Written {outfile}')
