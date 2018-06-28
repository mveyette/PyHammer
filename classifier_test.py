import classifier

def main():
    c = classifier.NNClassifier(hidden_units=[128,128,128], dropout=0.2)
    c.train(steps=2e3)
    

if __name__ == '__main__':
    main()
