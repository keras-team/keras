from keras.preprocessing import Image
train_datagen = Image.ImageDataGenerator(<>)

training_generator = training_data_generator.flow_from_directory(
training_data_dir,
<>)
x,y = train_generator.next() # x train, y train
# access x,y here in loop and do whatever you need
# Similar procedure with validation
# https://keras.io/api/preprocessing/image/
# https://www.geeksforgeeks.org/python-next-method/
