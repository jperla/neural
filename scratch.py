import os
import math
import datetime
import itertools
from itertools import izip, islice

import random
import Image
import numpy
import scipy
from scipy import signal

image_filename = 'textures/IMG_3756.JPG'

try:
    print full_image
except:
    pil_image = Image.open(image_filename).convert('L')
    full_image = numpy.asarray(pil_image).astype('double')
    small_image = full_image[600:1368,600:1624] #take middle of image



def gaussian(sigma=0.5, shape=None):
    """
    Gaussian kernel numpy array with given sigma and shape.

    The shape argument defaults to ceil(6*sigma).
    """
    sigma = numpy.max(abs(sigma), 1e-10)
    if shape is None:
        shape = numpy.max(int(6*sigma+0.5), 1)
    if not isinstance(shape, tuple):
        shape = (shape, shape)
    x = numpy.arange(-(shape[0]-1)/2.0, (shape[0]-1)/2.0+1e-8)
    y = numpy.arange(-(shape[1]-1)/2.0, (shape[1]-1)/2.0+1e-8)
    Kx = numpy.exp(-x**2/(2*sigma**2))
    Ky = numpy.exp(-y**2/(2*sigma**2))
    ans = numpy.outer(Kx, Ky) / (2.0*numpy.pi*sigma**2)
    return ans/sum(sum(ans))

def normalize_kernel(k):
    a = normalize_array(k, 1.0)
    s = float(sum(a.flatten()))
    area_1 = a / s
    assert (1.0 - float(sum(a.flatten())) < .1)
    return area_1


def theta_of_pixel(middle, i, j):
    i, j = (i - middle[0]), (j - middle[1])
    if i == 0 and j == 0:
        return 0
    else:
        theta = numpy.arctan2(-float(i), j)
        return theta

def dog_kernel(small_sigma, interval=numpy.sqrt(2)):
    big_sigma = small_sigma * interval
    shape = math.ceil(big_sigma * 10)
    inner = (gaussian(small_sigma, shape))
    outer = (gaussian(big_sigma, shape))
    norm_dog = normalize_kernel(outer) - normalize_kernel(inner)
    #array_to_file(outer, 'output/%s_outer_dog.png' % name)
    #array_to_file(inner, 'output/%s_inner_dog.png' % name)
    #array_to_file(norm_dog, 'output/%s_dog.png' % name)
    return norm_dog

def dog_kernel_radial(small_sigma, k=4, interval=numpy.sqrt(2), sin=False):
    dog = dog_kernel(small_sigma, interval)
    a, b = dog.shape
    middle = a / 2, b / 2
    for (i,j),v in numpy.ndenumerate(dog):
        theta = theta_of_pixel(middle, i, j)
        if theta is not None:
            if sin:
                dog[i,j] = v * numpy.sin(k * theta)
            else:
                dog[i,j] = v * numpy.cos(k * theta)
    return dog
    
def array_to_file(a, filename):
    a = normalize_array(a)
    i = Image.fromarray(a.astype('uint8'))
    return i.save(filename)

def normalize_array(a, norm_max=255):
    min = numpy.min(a.flatten())
    norm = numpy.max(a.flatten()) - min
    normed = norm_max * a / norm
    min = numpy.min(normed.flatten())
    centered = normed - min
    return centered

'''
# debug image saving
csb_pil = Image.open('csbldg.jpg').convert('L')
csb = numpy.asarray(csb_pil).astype('double')
filename = 'output/csb.jpg'
i = Image.fromarray(csb.astype('uint8'))
i.save(filename)
'''

'''
'''
# debug kernels:
try:
    print g
except:
    g = gaussian(10, 100)
    array_to_file (normalize_array(g), 'output/gaussian.png')
    dog = normalize_kernel(dog_kernel(2, interval=1.1))
    array_to_file (normalize_array(dog), 'output/dog.png')
    rdog = normalize_kernel(dog_kernel_radial(20, interval=1.1))
    array_to_file (normalize_array(rdog), 'output/rdog.png')


def run_convolve(i1, i2, mode='same'):
    print datetime.datetime.now()
    print 'convolving...'
    c = scipy.signal.fftconvolve(i1, i2, mode=mode)
    assert(i1.shape == c.shape)
    print 'done...'
    print datetime.datetime.now()
    return c

try:
    print c
except:
    print datetime.datetime.now()
    print 'convolving...'
    c = scipy.signal.fftconvolve(small_image, dog)
    print 'done...'
    print datetime.datetime.now()

def quantize_image(image, bins=5, edges=0):
    ''' Thresholds top and bottom of N bins '''
    _,bins = numpy.histogram(image[edges:-edges,edges:-edges].flatten(), bins=bins)
    def quantize(v, bins):
        for i,b in enumerate(bins):
            if v < b:
                return bins[i-1]
    for (i,j),v in numpy.ndenumerate(image):
        image[i,j] = quantize(v, bins)
    return image

def smooth_illuminance(image, sigma=50):
    g = normalize_kernel(gaussian(sigma))
    c = run_convolve(image, g, 'same')
    return image - c

def threshold_image(i, direction='max', bins=5, edges=0):
    ''' Thresholds top and bottom of N bins '''
    _,bins = numpy.histogram(i[edges:-edges,edges:-edges].flatten(), bins=bins)
    min_, max_ = bins[0], bins[-1]
    low, high = bins[1], bins[-2]
    if direction == 'max':
        t = numpy.where((high <= i), i, min_)
        t = zero_out_edges(t, edges, min_)
    else:
        t = numpy.where((i <= low), -i, -max_)
        t = zero_out_edges(t, edges, -max_)
    return t

def zero_out_edges(a, edges, zero):
    if edges > 0:
        a[:edges,:] = numpy.zeros([edges, a.shape[1]]) + zero
        a[-edges:,:] = numpy.zeros([edges, a.shape[1]]) + zero
        a[:,:edges] = numpy.zeros([a.shape[0], edges]) + zero
        a[:,-edges:] = numpy.zeros([a.shape[0], edges]) + zero
    return a



def add_noise(image, amount=.1):
    ''' Amount is a fraction of 1.  Default is 10% image noise'''
    noisy_image = numpy.array(image)
    high,low = numpy.max(image.flatten()), numpy.min(image.flatten())
    for (i,j),v in numpy.ndenumerate(noisy_image):

        noise = amount * (high - low) * ((random.random() * 2) - 1)
        #TODO: jperla: prefer gaussian normal noise
        #noise = numpy.random.normal(0, amount * (high - low))
        #noisy_image[i,j] = max(low, min(high, v + noise)) #more blacks and whites
        noisy_image[i,j] = v + noise
    return noisy_image

'''
array_to_file(add_noise(small_image, amount=.1), 'output/noisy_image_10.png')
array_to_file(add_noise(small_image, amount=0.2), 'output/noisy_image_20.png')
array_to_file(add_noise(small_image, amount=.5), 'output/noisy_image_50.png')
array_to_file(add_noise(small_image, amount=0.75), 'output/noisy_image_75.png')
array_to_file(add_noise(small_image, amount=1.0), 'output/noisy_image_100.png')
'''

'''
r = scipy.misc.imrotate(small_image, 10).astype('float64')
array_to_file(r, 'output/rotated_image_10.png')
r = scipy.misc.imrotate(small_image, 30).astype('float64')
array_to_file(r, 'output/rotated_image_30.png')
r = scipy.misc.imrotate(small_image, 45).astype('float64')
array_to_file(r, 'output/rotated_image_60.png')
r = scipy.misc.imrotate(small_image, 60).astype('float64')
array_to_file(r, 'output/rotated_image_60.png')
r = scipy.misc.imrotate(small_image, 74).astype('float64')
array_to_file(r, 'output/rotated_image_74.png')
r = scipy.misc.imrotate(small_image, 90).astype('float64')
array_to_file(r, 'output/rotated_image_90.png')
r = scipy.misc.imrotate(small_image, 180).astype('float64')
array_to_file(r, 'output/rotated_image_180.png')
r = scipy.misc.imrotate(small_image, 360).astype('float64')
array_to_file(r, 'output/rotated_image_360.png')
'''

try:
    print convolved
except:
    array_to_file(small_image, 'output/small_image.png')
    '''
    for i in xrange(1, 8):
        sigma = float(2)**float(i)
        k = cached_image_array('dog_%s' % sigma, lambda: dog_kernel(sigma))
        convolved = run_convolve(small_image, k)
        array_to_file(convolved, 'output/small_convolved_%s.png' % (2**i))
        sigma = (float(2)**float(i)) * math.sqrt(2)
        k = cached_image_array('dog_%s' % sigma, lambda: dog_kernel(sigma))
        convolved = run_convolve(small_image, k)
        array_to_file(convolved, 'output/small_convolved_%sr2.png' % (2**i))

    for i in xrange(12, 28):
        sigma = i
        k = cached_image_array('dog_%s' % sigma, lambda: dog_kernel(sigma))
        convolved = run_convolve(small_image, k)
        array_to_file(convolved, 'output/small_convolved_%s.png' % sigma)
    '''

def find_blob_centers(image):
    centers = []
    min = numpy.min(image.flatten())
    #TODO: jperla: i think this is pretty naive
    for (i,j),v in numpy.ndenumerate(image):
        #TODO: jperla: ignores edges
        if 1 < i < (image.shape[0] - 1) and 1 < j < (image.shape[1] - 1):
            if (v > min and v == numpy.max(image[i-1:i+2,j-1:j+2])):
                centers.append((i, j))
    return centers

def draw_blob_centers(image, centers, type='small'):
    image = numpy.array(image)
    max, min = numpy.max(image.flatten()), numpy.min(image.flatten())
    max = min + 1 if max == min else max
    dot = numpy.array([
                        [max, max, max, max, max, max, max],
                        [max, min, min, min, min, min, max],
                        [max, min, max, max, max, min, max],
                        [max, min, max, min, max, min, max],
                        [max, min, max, max, max, min, max],
                        [max, min, min, min, min, min, max],
                        [max, max, max, max, max, max, max],
                        ])
    dot = numpy.array([
                        [min, min, min, min, min, min, min, min, min],
                        [min, max, max, max, max, max, max, max, min],
                        [min, max, min, min, min, min, min, max, min],
                        [min, max, min, max, max, max, min, max, min],
                        [min, max, min, max, min, max, min, max, min],
                        [min, max, min, max, max, max, min, max, min],
                        [min, max, min, min, min, min, min, max, min],
                        [min, max, max, max, max, max, max, max, min],
                        [min, min, min, min, min, min, min, min, min],
                        ])
    if type != 'gray':
        dot = numpy.array([
                        [min, min, min, min, min, min, min, min, min],
                        [min, min, min, min, min, min, min, min, min],
                        [min, min, max, max, max, max, max, min, min],
                        [min, min, max, max, max, max, max, min, min],
                        [min, min, max, max, min, max, max, min, min],
                        [min, min, max, max, max, max, max, min, min],
                        [min, min, max, max, max, max, max, min, min],
                        [min, min, min, min, min, min, min, min, min],
                        [min, min, min, min, min, min, min, min, min],
                        ])
        dot = numpy.array([[max, max, max],
                            [max, min, max],
                            [max, max, max]])
    assert(dot.shape[0] % 2 == 1) # must be odd for calculations below
    radius = (dot.shape[0] / 2)
    for c in centers:
        i,j = c
        #TODO: jperla: naive ignore edges
        if radius < i < (image.shape[0] - radius) and \
            radius < j < (image.shape[1] - radius):
            image[i-radius:i+1+radius,j-radius:j+1+radius] = dot
    return image
    

def distance(a, b):
    return numpy.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def nearest_neighbors(center, centers, k=3):
    ''' nlogn can it be faster in batch? voronoi? '''
    #assert len(centers) > k
    nearest = sorted([(distance(center, c), tuple(c)) for c in centers if distance(c, center) > .0001])[:k]
    return [n for d,n in nearest[:k]]

def center_feature(center, centers):
    neighbors = nearest_neighbors(center, centers, 3)
    feature = [distance(center, n) for n in neighbors]
    return feature

def center_radial_feature(center, centers):
    neighbors = nearest_neighbors(center, centers, 3)
    nearest = neighbors[0]
    new_base = theta_of_pixel(center, nearest[0], nearest[1])
    features = []
    for n in neighbors:
        f = ((theta_of_pixel(center, n[0], n[1]) - new_base), distance(center, n))
        features.append(f)
    sorted_radially = list(itertools.chain(*sorted(features)))
    return sorted_radially

def center_neighbors_descriptor(center, centers):
    neighbors = [n for n in nearest_neighbors(center, centers, 30) if distance(center, n) < 100]
    #neighbors = nearest_neighbors(center, centers, 5)
    if len(neighbors) == 0:
        return []
    else:
        nearest = neighbors[0]
        new_base = theta_of_pixel(center, nearest[0], nearest[1])
        features = []
        for n in neighbors:
            f = ((theta_of_pixel(center, n[0], n[1]) - new_base), distance(center, n))
            features.append(f)
        return features

def distance_between_vector(a,b):
    return math.sqrt(sum((numpy.array(a) - numpy.array(b))**2))

def distance_between_all(v, rest):
    return sorted([distance_between_vector(v, r) for r in rest if sum(v - r) != 0])

def distance_between_radial(a, b):
    total = 0.0
    for p1,p2 in izip(slicen(a), slicen(b)):
        total += distance_between_polar(p1, p2)
    return total

def distance_between_polar(p1, p2):
    theta1, d1 = p1
    x1, y1 = d1 * math.sin(theta1), d1 * math.sin(theta1)
    theta2, d2 = p2
    x2, y2 = d2 * math.sin(theta1), d2 * math.sin(theta2)
    return distance((x1, y1), (x2, y2))
    

def slicen(s):
    assert(len(s) % 2 == 0)
    for (op, code) in izip(islice(s, 0, None, 2), islice(s, 1, None, 2)):
        yield op, code

def draw_image_blobs(name, image, kernel_name, kernel):
    convolved_name = '%s__c_%s' % (name, kernel_name)
    convolved  = cached_image_array(convolved_name,
                                lambda: run_convolve(image, kernel, mode='same'))

    edge_threshold = (kernel.shape[0] / 2) + 1
    centers_name = '%s_centers_%s.array' % (convolved_name, edge_threshold)
    centers = cached_array(centers_name,
                            lambda: calculate_threshold(name, 
                                                        convolved,
                                                        edge_threshold))

    descriptors = [center_neighbors_descriptor(c, centers) for c in centers]
    return centers, descriptors
                            
def calculate_threshold(name, convolved, edge_threshold):
    t = threshold_image(convolved, 'max', edges=edge_threshold)
    array_to_file(t, 'output/%s_thresholded_high.png' % name)
    high_centers = find_blob_centers(t)
    t = threshold_image(convolved, 'min', edges=edge_threshold)
    array_to_file(t, 'output/%s_thresholded_low.png' % name)
    low_centers = find_blob_centers(t)

    centers = high_centers + low_centers
    drawn = draw_blob_centers(numpy.zeros(t.shape), centers)
    array_to_file(drawn, 'output/%s_blobs.png' % name)

    drawn = draw_blob_centers(image, centers)
    array_to_file(drawn, 'output/%s_image.png' % name)
    return numpy.array(centers)

def transform_image(image, scrub):
    blur, rotation, noise = scrub
    #TODO: jperla: add blur?
    noised = add_noise(image, amount=noise)
    rotated = scipy.misc.imrotate(noised, rotation).astype('float64')
    #new = normalize_array(rotated, norm_max=1.0)
    new = rotated
    return new

def predict_location(f, original, scrub):
    blur, rotation, noise = scrub
    rotation = rotation * math.pi / 180.0 # to radians
    i, j = f
    middle = middle_of_image(original)
    theta = theta_of_pixel(middle, i, j)
    d = distance(middle, (i,j))
    assert(middle[0] - (math.sin(theta) * d) - i < .1)
    assert(middle[1] + (math.cos(theta) * d) - j < .1)
    return ((middle[0] - (d * math.sin(theta + rotation))), (middle[1] + (d * math.cos(theta + rotation))))

def crop_middle_bounds(image):
    s = image.shape
    m = numpy.min(s[0], s[1])
    size = (m / 2, m / 2)
    start = ((s[0] - size[0]) / 2, (s[1] - size[1]) / 2)
    return start, (start[0] + size[0], start[1] + size[1])

def move_cropped_point(crop, point):
    new_point = (point[0] - crop[0][0], point[1] - crop[0][1])
    return new_point

def crop_given_image(i, crop):
    return i[crop[0][0]:crop[1][0],crop[0][1]:crop[1][1]]

def calculate_feature_errors(name, image, scrub, kernel_name, kernel):
    transformed_name = '%s_z_%s' % (name, scrub)
    transformed = cached_image_array(transformed_name,
                                lambda: transform_image(image, scrub))
    crop = crop_middle_bounds(image)

    feature_locations, features = draw_image_blobs(name, image, kernel_name, kernel)

    cropped_transformed = crop_given_image(transformed, crop)
    found_feature_locations, found_features = draw_image_blobs(transformed_name, cropped_transformed, kernel_name, kernel)

    predicted_feature_locations = [predict_location(f, image, scrub) for f in feature_locations]
    cropped_predicted_feature_locations = [move_cropped_point(crop,f) for f in predicted_feature_locations]

    drawn = draw_blob_centers(transformed, predicted_feature_locations, type='gray')
    array_to_file(drawn, 'output/%s_predicted_whole.png' % name)

    drawn = draw_blob_centers(cropped_transformed, cropped_predicted_feature_locations, type='gray')
    #drawn = draw_blob_centers(drawn, feature_locations, type='gray')
    array_to_file(drawn, 'output/%s_predicted_errors.png' % name)
    drawn = draw_blob_centers(drawn, found_feature_locations, type='small')
    array_to_file(drawn, 'output/%s_errors.png' % name)

    distances = []
    for p in cropped_predicted_feature_locations:
        if not out_of_bounds(p, cropped_transformed, kernel.shape[0] / 2 + 1):
            closest = find_closest_feature(p, found_feature_locations)
            distances.append((p, closest))

    bounded_cropped_predicted_feature_locations = [p for p in cropped_predicted_feature_locations if not out_of_bounds(p, cropped_transformed, kernel.shape[0] / 2 + 1)]

    descriptor_match = []
    found_descriptors = [(c, center_neighbors_descriptor(c, found_feature_locations)) for c in found_feature_locations]
    for p in bounded_cropped_predicted_feature_locations:
        if not out_of_bounds(p, cropped_transformed, kernel.shape[0] / 2 + 100):
            v = center_neighbors_descriptor(p, bounded_cropped_predicted_feature_locations)
            closest = find_closest_descriptor(v, found_descriptors)
            descriptor_match.append(((p, v), closest))
    matches = [(distance(d[0][0], d[1][1]),d) for d in descriptor_match]

    return distances, matches

def out_of_bounds(p, image, bound_crop):
    if bound_crop <= p[0] < (image.shape[0] - bound_crop) and \
        bound_crop <= p[1] < (image.shape[1] - bound_crop):
        return False
    else:
        return True

def find_closest_feature(p, features):
    distances = sorted([(distance_between_polar(p, r),r) for r in features])
    return distances[0]

def distance_between_descriptors(d1, d2):
    '''
    Assumes it gets the original descriptor first, d2 is noisier descriptor.
    '''
    total = 0
    for f in d1:
        distance,g = find_closest_feature(f, d2)
        total += distance
    return total

def find_closest_descriptor(d, descriptors):
    distances = sorted([(distance_between_descriptors(d, r), tuple(p), tuple(r)) for p,r in descriptors])
    return distances[0]

'''
name = 'radial'
image = small_image
image = full_image
scrub = (0, 10, .01)
sigma=32
k = 2
kernel_name = 'dog_rad_%s_%s' % (sigma, k)
kernel = cached_image_array(kernel_name,
                            lambda: dog_kernel_radial(sigma, k))


name = 'full'
sigma = 16
kernel_name = 'dog_%s' % sigma
kernel = cached_image_array(kernel_name,
                            lambda: dog_kernel(sigma))

full_done = calculate_feature_errors(name, image, scrub, kernel_name, kernel)
'''

'''
d = []
for a,b in itertools.combinations(rfeatures, 2):
    d += [distance_between_radial(a,b)]
a = sorted(d[:1000])
'''




'''
array_to_file(dog, 'output/dog.png')
array_to_file(c, 'output/small_convolved.png')
'''

def zernike(m, n, size):
    assert(n >= abs(m))
    if m >= 0:
        radial,m = numpy.cos, m
    else:
        radial,m = numpy.sin, -m
    z = numpy.zeros((size, size))
    if (n - m) % 2 == 0:
        poly = zernike_polynomial(m, n)
        a, b = z.shape
        middle = a / 2, b / 2
        for (i,j), v in numpy.ndenumerate(z):
            theta = theta_of_pixel(middle, i, j)
            if theta is not None:
                rho = distance(middle, (i,j)) / (size / 2)
                if rho <= 1.0:
                    z[i,j] = poly(rho) * radial(m * theta)
    return z

def zernike_polynomial(m, n):
    lower = (n - m) / 2
    higher = (n + m) / 2
    k = numpy.array(xrange(0, lower + 1))
    exponent = n - (2 * k)
    def crazy_frac(i):
        f = scipy.misc.factorial
        numerator = (((float(-1))**float(i)) * f(n - i))
        denominator = f(i) * f(higher - i) * f(lower - i)
        return float(numerator) / float(denominator)
    frac = numpy.array([crazy_frac(i) for i in k])
    assert(len(exponent) == len(frac))
    def poly(rho):
        s = sum([(frac[i] * (rho**(exponent[i]))) for i in xrange(len(frac))])
        return s
    return poly

def is_same_function(f1, f2):
    for x in xrange(-100, 100, 1):
        x = float(x) / 10
        if not (f1(x) - f2(x) < .1):
            return False
    else:
        return True

def middle_of_image(image):
    a, b = image.shape
    middle = a / 2, b / 2
    return middle

def semispherical_smooth(image):
    image = numpy.array(image)
    middle = middle_of_image(image)
    assert(image.shape[0] == image.shape[1])
    max_distance = image.shape[1] - middle[1]
    for (i,j),v in numpy.ndenumerate(image):
        rho = distance(middle, (i,j)) / max_distance
        if rho <= 1.0:
            image[i,j] = v * numpy.sqrt(1 - (rho ** 2))
    return image

'''
is_same_function(zernike_polynomial(0, 0), lambda x: 1.0)
is_same_function(zernike_polynomial(1, 1), lambda x: x)
is_same_function(zernike_polynomial(2, 2), lambda x: (2 * (x ** 2)))
is_same_function(zernike_polynomial(3, 3), lambda x: (x ** 3))
is_same_function(zernike_polynomial(4, 4), lambda x: (x ** 4))
is_same_function(zernike_polynomial(5, 5), lambda x: (x ** 5))
is_same_function(zernike_polynomial(6, 6), lambda x: (x ** 6))
is_same_function(zernike_polynomial(1, 3), lambda x: ((3 * (x ** 3)) - (2 * x)))
is_same_function(zernike_polynomial(1, 5), lambda x: ((10 * (x ** 5)) - (12 * (x ** 3)) + (3 * x)))
is_same_function(zernike_polynomial(0, 4), lambda x: ((6 * (x ** 4)) - (6 * (x ** 2)) + 1))
is_same_function(zernike_polynomial(2, 4), lambda x: ((4 * (x ** 4)) - (2 * (x ** 2))))
'''

'''
for n in xrange(0, 6):
    for m in xrange(-n, n+1):
        if (n - m) % 2 == 0:
            z = semispherical_smooth(zernike(m, n, 200))
            array_to_file(z, 'output/zernike_%s_%s.png' % (m, n))
'''

def cached_image_array(name, f):
    cache_dir = 'output/'
    array_path = os.path.join(cache_dir, '%s.array' % name)
    image_path = os.path.join(cache_dir, '%s.png' % name)
    
    if os.path.exists(array_path):
        output = numpy.genfromtxt(array_path, dtype='double')
    else:
        output = f()
        assert(isinstance(output, numpy.ndarray))
        numpy.savetxt(array_path, output)
        array_to_file(output, image_path)
    assert(os.path.exists(array_path))
    assert(os.path.exists(image_path))
    return output

def cached_array(name, f):
    cache_dir = 'output/'
    array_path = os.path.join(cache_dir, name)
    if os.path.exists(array_path):
        output = numpy.genfromtxt(array_path, dtype='double')
    else:
        output = f()
        assert(isinstance(output, numpy.ndarray))
        numpy.savetxt(array_path, output)
    assert(os.path.exists(array_path))
    return output

'''
def cache_array(name, f, cache_dir='cache'):
    path = os.path.join(cache_dir, name)
    if not os.path.exists(path):
        array = f()
        assert(array.dtype == 'double')
        array.tofile(path)
    return numpy.fromfile(path, dtype='double')
'''

datetime.datetime.now()
