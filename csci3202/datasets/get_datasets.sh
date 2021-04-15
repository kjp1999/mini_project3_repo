# Get CIFAR10

if [[ "$OSTYPE" == "darwin"* ]]; then # OS X
  curl http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -o cifar-10-python.tar.gz
else # Linux, Cygwin, BSD, ...
  wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
fi

tar -xzvf cifar-10-python.tar.gz
rm cifar-10-python.tar.gz 
