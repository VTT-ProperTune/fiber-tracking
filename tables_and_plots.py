import zipfile, pickle
import pandas as pd
import numpy as np
from math import *
from matplotlib import pyplot
import sys
import argparse

def get_orientation_tensor(tangents):
    norms = np.linalg.norm(tangents, axis=1)
    d = (tangents.T / norms).T
    return np.einsum('ij,il', d, d) / len(d)

def load_data(filename):
    with zipfile.ZipFile(filename) as zf:
        with zf.open('fibers.csv') as f:
            fiberdata = pd.read_csv(f)
        with zf.open('kink_angles.pickle') as f:
            kink_angles = pickle.load(f)
        with zf.open('segments.pickle') as f:
            segments = pickle.load(f)
            segment_lengths = [s[0] for s in segments]
            segment_shape_factors = [s[1] for s in segments]
    
    return fiberdata, kink_angles, segment_lengths, segment_shape_factors



def add_ct_data(table, data):
    fiberdata, kink_angles, segment_lengths, segment_shape_factors = data

    lengths = fiberdata['length'] / 1000
    extent_vectors = np.column_stack([fiberdata['extent vec x'], fiberdata['extent vec y'], fiberdata['extent vec z']]) / 1000
    shape_factors = np.linalg.norm(extent_vectors, axis=1)/lengths
    ML_factors = fiberdata['ML factor']
    interior_mask = fiberdata['interior flag'] > 0
    
    sel = lengths > 0.2

    # Exponential distribution is unchanged by clipping the left side and moving to start from origin...
    mean_length = np.average(lengths[sel*interior_mask], weights=ML_factors[sel*interior_mask]) - 0.2

    mean_shape_factor = np.average(shape_factors[sel*interior_mask], weights=ML_factors[sel*interior_mask])
    
    kink_numbers = np.array([len(k) for k in kink_angles])
    large_kink_numbers = np.array([np.count_nonzero(k>60) for k in kink_angles])
    kink_densities = kink_numbers/lengths
    large_kink_densities = large_kink_numbers/lengths

    kink_density = np.mean(kink_densities[sel])
    large_kink_density = np.mean(large_kink_densities[sel])
    mean_kink_angle = np.mean(np.concatenate([kink_angles[s] for s in np.flatnonzero(sel)]))

    mean_segment_length = np.mean(np.concatenate(segment_lengths)) / 1000
    
    table.setdefault('mean length', []).append(mean_length)
    table.setdefault('mean shape factor', []).append(mean_shape_factor)
    table.setdefault('mean kink angle', []).append(mean_kink_angle)
    table.setdefault('kink density', []).append(kink_density)
    table.setdefault('large kink density', []).append(large_kink_density)
    table.setdefault('mean segment length', []).append(mean_segment_length)


def plot_shape_factor(data):
    fiberdata, kink_angles, segment_lengths, segment_shape_factors = data
    lengths = fiberdata['length']
    extent_vectors = np.column_stack([fiberdata['extent vec x'], fiberdata['extent vec y'], fiberdata['extent vec z']])
    shape_factors = np.linalg.norm(extent_vectors, axis=1)/lengths
    
    max_class = np.round(np.max(lengths), decimals=-2)
    class_idxs = np.digitize(lengths, np.arange(0, max_class, 100))
    classes = [class_idxs==i for i in np.unique(class_idxs)]
    class_means = np.array([np.mean(lengths[cl]) for cl in classes])
    
    class_shape_factors = [np.mean(shape_factors[cl]) for cl in classes]
    
    sel = slice(2, -3)
    pyplot.plot(class_means[sel]/1000, class_shape_factors[sel], 'x--')
    pyplot.xlabel('Fiber length (mm)')
    pyplot.ylabel('Mean shape factor')
    
    
def kink_distribution_plot(data, large_kinks=False, **kwargs):
    fiberdata, kink_angles, segment_lengths, segment_shape_factors = data
    lengths = fiberdata['length']
    
    max_class = np.round(np.max(lengths), decimals=-2)
    class_idxs = np.digitize(lengths, np.arange(0, max_class, 100))
    classes = [class_idxs==i for i in np.unique(class_idxs)]
    class_means = np.array([np.mean(lengths[cl]) for cl in classes])
    
    kink_numbers = np.array([len(k) for k in kink_angles])
    large_kink_numbers = np.array([np.count_nonzero(k>60) for k in kink_angles])
    if large_kinks:
        kink_densities = large_kink_numbers/lengths
        pyplot.ylabel('Kink ($>60^\circ$) density (mm$^{-1}$)')
    else:
        kink_densities = kink_numbers/lengths
        pyplot.ylabel('Kink density (mm$^{-1}$)')
        
    class_kink_densities = [np.mean(kink_densities[cl])*1000 for cl in classes]
    class_kink_stds = [np.std(kink_densities[cl]) for cl in classes]

    expected_class_kink_stds = [sqrt(d/m) for m,d in zip(class_means, class_kink_densities)]
    
    sel_stop = next((i for i in range(len(classes)) if np.count_nonzero(classes[i])<8), None)
    sel = slice(2,sel_stop)
    line = pyplot.plot(class_means[sel]/1000, class_kink_densities[sel], **kwargs)
    pyplot.xlabel('Fibre length (mm)')
    return line


def orientation_plot(data, **kwargs):
    fiberdata = data[0]
 
    extent_vectors = np.column_stack([fiberdata['extent vec x'], fiberdata['extent vec y'], fiberdata['extent vec z']])
    
    lengths = fiberdata['length']
    max_class = np.round(np.max(lengths), decimals=-2)
    class_idxs = np.digitize(lengths, np.arange(0, max_class, 100))
    classes = [class_idxs==i for i in np.unique(class_idxs)]
    class_means = np.array([np.mean(lengths[cl]) for cl in classes])
    
    ori = get_orientation_tensor(extent_vectors)
    idx = np.argmax([ori[0,0], ori[1,1], ori[2,2]])
    
    class_oris = np.array([get_orientation_tensor(extent_vectors[cl])[idx,idx] for cl in classes])
    hermans_parameters = 1/2*(3*class_oris - 1)
    
    sel_stop = next((i for i in range(len(classes)) if np.count_nonzero(classes[i])<8), None)
    sel = slice(1,sel_stop)
    line = pyplot.plot(class_means[sel]/1000, hermans_parameters[sel], **kwargs)
    x = np.linspace(0, 3)
    pyplot.xlabel('Fibre length (mm)')
    pyplot.ylabel('Orientation parameter')
    return line


if __name__ == '__main__':
    # Example script to load a single datafile and print some statistical quantities
    # Above functions can be used to make some plots
    
    parser = argparse.ArgumentParser()
    parser.add_argument('analysis_file', help='The output file from fiber_analysis (.zip)')
    args = parser.parse_args()
    
    data = load_data(args.analysis_file)

    table = {}
    add_ct_data(table, data)
    table = pd.DataFrame(table)
    print(table)
