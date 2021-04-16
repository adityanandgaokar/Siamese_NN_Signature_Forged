
import matplotlib.pyplot as plt
from train import siamese_datagen
from keras.models import Model
import numpy as np

def compute_accuracy_roc(predictions, labels):
    '''Compute ROC accuracy with a range of thresholds on distances.
    '''
    dmax = np.max(predictions)
    dmin = np.min(predictions)
    nsame = np.sum(labels == 1)
    ndiff = np.sum(labels == 0)
   
    step = 0.01
    max_acc = 0
    best_thresh = -1
   
    for d in np.arange(dmin, dmax+step, step):
        idx1 = predictions.ravel() <= d
        idx2 = predictions.ravel() > d
       
        tpr = float(np.sum(labels[idx1] == 1)) / nsame       
        tnr = float(np.sum(labels[idx2] == 0)) / ndiff
        acc = 0.5 * (tpr + tnr)       
#       print ('ROC', acc, tpr, tnr)
       
        if (acc > max_acc):
            max_acc, best_thresh = acc, d
           
    return max_acc, best_thresh




img_h, img_w = 150, 200
image_shape = (img_h, img_w, 1)

featureExtractor = build_siamese_model(image_shape)
imgA = Input(shape = (image_shape))
imgB = Input(shape = (image_shape))
featsA = featureExtractor(imgA)
featsB = featureExtractor(imgB)

distance = Lambda(euclidean_distance, output_shape = euclidean_distance_output_shape)([featsA, featsB])

model = Model(inputs=[imgA, imgB], outputs=distance)  


model.load_weights('/content/siamese_network.h5')

test_gen = siamese_datagen(test_dir, test_csv, 1)
pred, tr_y = [], []



for i in range(validation_samples):
    test_point, test_label = next(test_gen)
    img1, img2 = test_point[0], test_point[1]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 10))
    ax1.imshow(np.squeeze(img1), cmap='gray')
    ax2.imshow(np.squeeze(img2), cmap='gray')
    ax1.set_title('Genuine')
    if test_label == 0:
        ax2.set_title('Genuine')
    else:
        ax2.set_title('Forged')
    ax1.axis('off')
    ax2.axis('off')
    plt.show()
    result = model.predict([img1, img2])
    diff = result[0][0]
    print("Difference Score = ", diff)
    if diff > 0.10:
        print("Its a Forged Signature")
    else:
        print("Its a Genuine Signature")



