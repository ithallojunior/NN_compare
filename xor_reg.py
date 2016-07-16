#Example  with a Regressor using the scikit-learn library
# example for the XOr gate
from sklearn.neural_network import MLPRegressor 

X = [[0., 0.],[0., 1.], [1., 0.], [1., 1.]] # each one of the entries 00 01 10 11
y = [0, 1, 1, 0] # outputs for each one of the entries

# check http://scikit-learn.org/dev/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor
#for more details
reg = MLPRegressor(hidden_layer_sizes=(5),activation='tanh', algorithm='sgd', alpha=0.001, learning_rate='constant',
                   max_iter=10000, random_state=None, verbose=False, warm_start=False, momentum=0.8, tol=10e-8, shuffle=False)

reg.fit(X,y)

outp =  reg.predict([[0., 0.],[0., 1.], [1., 0.], [1., 1.]])

print'Results:'
print '0 0 0:', outp[0]
print '0 1 1:', outp[1]
print '1 0 1:', outp[2]
print '1 1 0:', outp[0]
print'Score:', reg.score(X, y)