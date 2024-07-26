#import libraries needed 
import numpy as np
from mygrad import Tensor
from mynn.layers.dense import dense
from mynn.optimizers.sgd import SGD
from mygrad.nnet.initializers import glorot_normal
from mygrad.nnet.losses import margin_ranking_loss
import pickle 

#created myNN model for embedding image descriptors
class ImageDescriptors:
    def __init__(self, caption_dimension, image_dimension, embedding_dimension):
        '''Initializes all of the layers in our model, and sets them
        as attributes of the model.
        
        Parameters
        ----------
        caption_dimension : int
            The size of the captions.
            
        image_dimension : int
            The size of the images
        
        embedding_dimension : int
            The size of the embedded descriptor
        '''
        self.caption_embed = dense(caption_dimension, embedding_dimension, weight_initializer = glorot_normal, bias = False)
        self.image_embed = dense(image_dimension, embedding_dimension, weight_initializer = glorot_normal, bias = False)
        
    def __call__(self, caption, image):
        '''Passes data as input to our model, forword pass.
        
        Parameters
        ----------
        caption : Union[numpy.ndarray, mygrad.Tensor], shape=(M, caption_dimension)
            A batch of caption data consisting of M captions,
            each with a dimensionality of caption_dim.

        image : Union[numpy.ndarray, mygrad.Tensor], shape=(M, image_dimension)
            A batch of image data consisting of M images,
            each with a dimensionality of image_dim.

        Returns
        -------
        Tuple[mygrad.Tensor, mygrad.Tensor], each shape=(M, embedding_dim)
        '''
        caption_emb = self.caption_embed(caption)
        image_emb = self.image_embed(image)
        return caption_emb, image_emb
    def parameters(self):
        '''A convenience function for getting all the parameters of our model.
        
        This can be accessed as an attribute, via `model.parameters` 
        
        Returns
        -------
        Tuple[Tensor, ...]
            A tuple containing all of the learnable parameters for our model
        '''
        return self.caption_embed.parameters + self.image_embed.parameters

#extracting sets of data (caption-ID, image-ID, confusor-image-ID) triples, creating training and validation sets 
def shuffle_and_split_data(data, validation_split=0.2):
    '''Shuffles the data and splits it into training and validation sets.
    
    Parameters
    ----------
    data : numpy.ndarray, shape=(M, N)
        Array of generic data.
        
    Returns
    -------
    Tuple of numpy.ndarrays
        Training and validation sets for captions, images, and confusors will be split and created.
    '''
    data -= np.mean(data, axis = 0)
    data /= np.std(data, axis = 0)
    N = data.shape[0]
    indices = np.arange(N)
    np.random.shuffle(indices)
    data = data[indices]
    split_idx = int(N * (1 - validation_split))
    train_data = data[:split_idx]
    validation_data = data[split_idx:]
    return train_data, validation_data
def extract_data(caption, image, confusor):
    '''Extract training and validation sets from three arrays needed to train model
    
    Parameters
    ----------
    captions : numpy.ndarray, shape=(N, caption_dim)
        Array of caption data.
        
    images : numpy.ndarray, shape=(N, image_dim)
        Array of image data.
        
    confusors : numpy.ndarray, shape=(N, image_dim)
        Array of confusor image data.
        
    Returns
    -------
    Tuple of numpy.ndarrays
        Training and validation sets for captions, images, and confusors.
    '''
    training_set = (shuffle_and_split_data(caption)[0], shuffle_and_split_data(image)[0], shuffle_and_split_data(confusor)[0])
    validation_set = (shuffle_and_split_data(caption)[1], shuffle_and_split_data(image)[1], shuffle_and_split_data(confusor)[1])
    return training_set, validation_set

#function to compute loss (mygrad's margin ranking loss) and accuracy (fraction of dot product pairs satisfy image_true * caption > image_confusor * caption)
def compute_loss_and_accuracy(caption_emb, image_emb, confusor_emb):
    '''Compute margin ranking loss and accuracy.
    
    Parameters
    ----------
    caption_emb : mygrad.Tensor, shape=(M, embedding_dim)
        Embeddings for the captions.
    
    image_emb : mygrad.Tensor, shape=(M, embedding_dim)
        Embeddings for the true images.
    
    confusor_emb : mygrad.Tensor, shape=(M, embedding_dim)
        Embeddings for the confusor images.
    
    Returns
    -------
    loss : mygrad.Tensor
        The computed margin ranking loss.
    
    accuracy : float
        The fraction of correct pairs where the similarity score of the correct image is higher than that of the confusor image.
    '''
    good_sim = (caption_emb @ image_emb.T)
    bad_sim = (caption_emb @ confusor_emb.T)
    
    loss = margin_ranking_loss(x1 = good_sim, x2 = bad_sim, y = 1, margin = 0.25)
    accuracy = np.mean(good_sim > bad_sim)
    
    return loss, accuracy

def save_model(model, file_path):
    '''Saves the model weights to a file.
    
    Parameters
    ----------
    model : ImageDescriptors
        Trained model containing weights
        
    file_path : str
        Path for the model's weights will be saved
    '''
    with open(file_path, 'wb') as f:
        pickle.dump({k: v.data for k, v in model.parameters.items()}, f)

def load_model(model, file_path):
    '''Loads the model weights from a file.
    
    Parameters
    ----------
    model : ImageDescriptors
        Model where weights will load
        
    file_path : str
       Path where model's weights will be loaded
    '''
    with open(file_path, 'rb') as f:
        weights = pickle.load(f)

    for k, v in weights.items():
        model.parameters[k].data = v


#model and optimizer are initialized
model = ImageDescriptors(caption_dimension = 200, image_dimension = 512, embedding_dimension = 128)
optimizer = SGD(model.parameters, learning_rate = 1e-3, momentum = 0.9)
batch_size = 32
num_epochs = 500

#training set data is extracted, need to get captions, image, confusors from dataset and pass through this function for model to work
training_set = extract_data(captions, images, confusors)

for epoch in range(num_epochs):
    total_loss, total_accuracy = 0.0

    # Shuffle the data at the beginning of each epoch
    indices = np.arange(len(training_set[0]))
    np.random.shuffle(indices)
    training_set = (training_set[0][indices], training_set[1][indices], training_set[2][indices])

    #process batches of data
    for i in range(0, len(training_set[0]), batch_size):
        batch_captions = Tensor(training_set[0][i : i + batch_size])
        batch_images = Tensor(training_set[1][i : i + batch_size])
        batch_confusors = Tensor(training_set[2][i : i + batch_size])

        #perform forword pass
        caption_emb, image_emb = model(batch_captions, batch_images)
        _, confusor_emb = model(batch_captions, batch_confusors)

        #loss and accuracy computation
        loss, accuracy = compute_loss_and_accuracy(caption_emb, image_emb, confusor_emb)

        #optimize
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss = total_loss + loss.item() * len(batch_captions)
        total_accuracy = total_accuracy + accuracy * len(batch_captions)

    avg_loss = total_loss / len(training_set[0])
    avg_accuracy = total_accuracy / len(training_set[0])

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

#call save and load functions to determine final model parameters
save_model(model, 'mynn_model_weights_final.pkl')
load_model(model, 'mynn_model_weights_final.pkl')
