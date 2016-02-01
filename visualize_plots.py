def figures(history,figure_name="plots"):
    """ method to visualize accuracies and loss vs epoch for training as well as testind data\n
        Argumets: history     = an instance returned by model.fit method\n
                  figure_name = a string representing file name to plots. By default it is set to "plots" \n
       Usage: hist = model.fit(X,y)\n              figures(hist) """
    from keras.callbacks import History
    if isinstance(history,History):
        import matplotlib.pyplot as plt
        hist     = history.history 
        epoch    = history.epoch
        acc      = hist['acc']
        loss     = hist['loss']
        val_loss = hist['val_loss']
        val_acc  = hist['val_acc']
        plt.figure(1)

        plt.subplot(221)
        plt.plot(epoch,acc)
        plt.title("Training accuracy vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")     

        plt.subplot(222)
        plt.plot(epoch,loss)
        plt.title("Training loss vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")  

        plt.subplot(223)
        plt.plot(epoch,val_acc)
        plt.title("Validation Acc vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Accuracy")  

        plt.subplot(224)
        plt.plot(epoch,val_loss)
        plt.title("Validation loss vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss")  
        plt.tight_layout()
	plt.savefig(figure_name)
    else:
        print "Input Argument is not an instance of class History"
