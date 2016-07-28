#Example with a Classifier using the scikit-learn library
#example for the XOr gate
#It shows how to save the network to a file too


from sklearn.neural_network import MLPClassifier
import pickle

X = [[0., 0.],[0., 1.], [1., 0.], [1., 1.]] # each one of the entries 00 01 10 11
y = [0, 1, 1, 0] # outputs for each one of the entries

clf = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(10), random_state=1, verbose=True, max_iter=1000)
clf.fit(X, y)

outp =  clf.predict([[0., 0.],[0., 1.], [1., 0.], [1., 1.]])
print'Results:'
print '0 0:', outp[0]
print '0 1:', outp[1]
print '1 0:', outp[2]
print '1 1:', outp[3]
print'Score:', clf.score(X, y)

#saving into file for later usage
f = open('data.ann', 'r+w')
mem = pickle.dumps(clf)
f.write(mem)
f.close()
f = open('data.ann', 'r+w')
nw = f.read()
mem = pickle.loads(nw)
mem.predict([[1., 1.],[1., 0.], [0., 1.], [0., 0.]])

print'Results:'
print '0 0:', outp[3]
print '0 1:', outp[2]
print '1 0:', outp[1]
print '1 1:', outp[0]
print'Score:', clf.score(X, y)
