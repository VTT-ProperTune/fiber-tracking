from skimage import io, morphology, segmentation, measure
from scipy import ndimage, signal, stats
import numpy as np
from math import *
from matplotlib import pyplot
import pandas as pd
import zipfile, pickle
import sys
import argparse

try:
    from fiber_tracking import load_results, find_path_volumes, spline_fit, draw_spline, DirectedPath, spline_arclength
except:
    print('Could not import fiber_tracking. Consider running with "PYTHONPATH=/path/to/code/folder python fiber_analysis.py"')


def discretize_spline(spline, length=None, ds=1.0):
    if length is None:
        length = spline_arclength(spline)
    n = int(np.ceil(length/ds))
    t = np.linspace(0, 1, n)
    pts = spline(t)
    return pts

def calc_principal_axes(points, return_moments=False):
	points = points - np.mean(points, axis=0)
	X, Y, Z = points.T
	
	Ixx = np.sum(Y*Y + Z*Z)
	Iyy = np.sum(Z*Z + X*X)
	Izz = np.sum(X*X + Y*Y)
	Ixy = -np.sum(X*Y)
	Iyz = -np.sum(Y*Z)
	Izx = -np.sum(Z*X)
	
	I = np.array([[Ixx, Ixy, Izx], [Ixy, Iyy, Iyz], [Izx, Iyz, Izz]])
	principal_moments, principal_axes = np.linalg.eigh(I)
	if not return_moments:
		return principal_axes.T
	else:
		return principal_axes.T, principal_moments

def spline_cum_arclength(c, t=np.linspace(0,1,100)):
    """ Arc length of a spline """
    tangent = c.derivative()
    dt = t[1]
    vals = np.linalg.norm(tangent(t), axis=1)
    integral = np.cumsum((vals[1:]+vals[:-1])*(dt/2.0))
    return np.r_[0.0, integral]

def get_spline_extent(spline):
    pts = discretize_spline(spline)
    ax = calc_principal_axes(pts)[0]
    proj = np.dot(pts, ax)
    extent = np.max(proj) - np.min(proj)
    return extent*ax


def get_tangents(spline, ds=1.0):
    length = spline_arclength(spline)
    n = int(np.ceil(length/ds))
    t = np.linspace(0, 1, n)
    tangents = spline.derivative()(t)
    return tangents
    
def calc_turn_angles(spline, h, ds=1.0):
    length = spline_arclength(spline)
    n = int(np.ceil(length/ds))
    t = np.linspace(0, 1, n)
    tangents = spline.derivative()(t)
    tangents = (tangents.T/np.linalg.norm(tangents, axis=1)).T
    shift = round(h/ds)
    angles = np.arccos(np.sum(tangents[shift:]*tangents[:-shift], axis=1))
    s = shift//2
    return np.r_[[np.nan]*s, angles, [np.nan]*(shift-s)]


def find_kinks_and_segments(spline, h, threshold=40, min_spacing=None):
    angles = calc_turn_angles(spline, h)*180/pi
    positions, info = signal.find_peaks(angles, height=threshold)
    heights = info['peak_heights']
    n = len(angles)
    cumlength = spline_cum_arclength(spline, np.linspace(0,1,n))
        
    if min_spacing is not None and len(positions) > 1:
        # Note: min_spacing is in voxel units
        order = np.argsort(heights)[::-1]
        accepted = [order[0]]
        for i in order[1:]:
            p1 = cumlength[positions[i]]
            if all(abs(p1 - cumlength[positions[j]]) > min_spacing for j in accepted):
                accepted.append(i)
        positions = positions[accepted]
        heights = heights[accepted]
        order = np.argsort(positions)
        positions = positions[order]
        heights = heights[order]
        
    segment_endpoints = spline(np.r_[0, positions/(n-1), 1])
    e2e_distances = np.linalg.norm(np.diff(segment_endpoints, axis=0), axis=1)
    lengths = np.diff(np.r_[0, cumlength[positions], cumlength[-1]])
    shape_factors = e2e_distances/lengths
    
    return heights, lengths, shape_factors


def is_interior_fiber(voxels, coords, img_shape, margin):
    shape = np.array(img_shape)
    pts = coords[voxels[[0,-1]]]
    cmin = np.min(pts, axis=0)
    cmax = np.max(pts, axis=0)
    return np.all(cmin>margin) and np.all(shape-1 - cmax > margin)



def miles_lantuejoul(voxels, coords, img_shape, r_bar):
    pts = coords[voxels]
    bbmin = np.min(pts, axis=0) - r_bar/2
    bbmax = np.max(pts, axis=0) + r_bar/2
    lz, ly, lx = img_shape
    bbz, bby, bbx = bbmax - bbmin
    return (lx*ly*lz)/((lx-bbx)*(ly-bby)*(lz-bbz))


def simulate_kink_distribution(lengths, kink_density):
    kink_numbers = np.array([stats.poisson.rvs(l*kink_density) for l in lengths])
    return kink_numbers



def get_orientation_tensor(tangents, weights=None):
    norms = np.linalg.norm(tangents, axis=1)
    d = (tangents.T / norms).T
    if weights == None: weights = np.ones(len(tangents))
    return np.einsum('ij,il,i', d, d, weights) / np.sum(weights)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tracking_results_file', help='The output file from fiber tracking (.zip)')
    parser.add_argument('image_file', help='Binarized image from which skeleton was calculated')
    parser.add_argument('--output_filename', default=None, help='Output filename with .zip extension (otherwise default is used)')
    args = parser.parse_args()
    
    analysis_file = args.tracking_results_file
    img_file = args.image_file
    if args.output_filename is None:
        output_filename = analysis_file.replace('.zip', '_fiberdata.zip')
    else:
        output_filename = args.output_filename
    
    print('Loading data')
    system = load_results(analysis_file)
    img = io.imread(img_file)

    r_bar = system.r_bar
    spacing = system.spacing[0]

    print('Fitting splines')
    splines = []
    fiber_voxels = []
    for f in system.fibers:
        voxels = []
        for i,p in enumerate(f):
            s = 1 if i != 0 else 0
            e = -1 if i != len(f)-1 else None
            voxels.append(p.voxels[s:e])
        voxels = np.concatenate(voxels)
        fiber_voxels.append(voxels)
        tot_length = sum(p.length for p in f)
        coords = system.coords[voxels]
        n_cpts = 2 + int(tot_length//(10*r_bar))
        spline = spline_fit(coords, 3, n_cpts)
        splines.append(spline)

    interior_mask = np.array([is_interior_fiber(v, system.coords, img.shape, r_bar*2) for v in fiber_voxels])
    ML_factors = np.array([miles_lantuejoul(v, system.coords, img.shape, r_bar) for v in fiber_voxels])

    print('Calculating length measures')
    lengths = np.array([spline_arclength(s) for s in splines])
    lengths_um = lengths * spacing

    extent_vectors = np.array([get_spline_extent(s) for s in splines])

    ori_tensors = np.array([get_orientation_tensor(get_tangents(s)) for s in splines])

    fiber_data = pd.DataFrame({
        'length': lengths_um,
        'extent vec x': extent_vectors[:,2]*spacing,
        'extent vec y': extent_vectors[:,1]*spacing,
        'extent vec z': extent_vectors[:,0]*spacing,
        'ML factor': ML_factors,
        'interior flag': interior_mask.astype(int),
        'ori xx': ori_tensors[:,2,2],
        'ori yy': ori_tensors[:,1,1],
        'ori zz': ori_tensors[:,0,0]})


    print('Finding kinks')

    tmp = [find_kinks_and_segments(s, 2*r_bar, 30, min_spacing=200/spacing) for s in splines]
    kink_angles = [v[0] for v in tmp]
    segment_lengths_um = [v[1]*spacing for v in tmp]
    segment_shape_factors = [v[2] for v in tmp]

    
    with zipfile.ZipFile(output_filename, mode='w', compression=zipfile.ZIP_DEFLATED) as f:
        with f.open('fibers.csv', 'w') as fiber_table_f:
            fiber_data.to_csv(fiber_table_f)
        with f.open('kink_angles.pickle', 'w') as kink_angles_f:
            pickle.dump(kink_angles, kink_angles_f)
        with f.open('segments.pickle', 'w') as segments_f:
            pickle.dump(list(zip(segment_lengths_um, segment_shape_factors)), segments_f)
    print(f'Written {output_filename}')
