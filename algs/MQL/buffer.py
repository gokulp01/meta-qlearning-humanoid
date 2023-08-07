import numpy as np

class Buffer(object):
	def __init__(self, max_size=1e6):

		self.storage = []
		self.max_size = max_size
		self.ptr = 0

	def reset(self):
		self.storage = []
		self.ptr = 0

	def add(self, data):
		if len(self.storage) == self.max_size:
			self.storage[int(self.ptr)] = data
			self.ptr = (self.ptr + 1) % self.max_size
		else:
			self.storage.append(data)

	def size_rb(self):
		if len(self.storage) == self.max_size:
			return self.max_size

		else:
			return len(self.storage)

	def sample(self, batch_size):
		ind = np.random.randint(0, len(self.storage), size=batch_size)
		x, y, u, r, d, pu, pr, px, nu, nr, nx = [], [], [], [], [], [], [], [], [], [], []

		for i in ind: 
			X, Y, U, R, D, PU, PR, PX, NU, NR, NX = self.storage[i]
			x.append(np.array(X, copy=False))
			y.append(np.array(Y, copy=False))
			u.append(np.array(U, copy=False))
			r.append(np.array(R, copy=False))
			d.append(np.array(D, copy=False))
			pu.append(np.array(PU, copy=False))
			pr.append(np.array(PR, copy=False))
			px.append(np.array(PX, copy=False))
			nu.append(np.array(NU, copy=False))
			nr.append(np.array(NR, copy=False))
			nx.append(np.array(NX, copy=False))

		return np.array(x), np.array(y), np.array(u), \
			   np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1), \
			   np.array(pu), np.array(pr), np.array(px),\
			   np.array(nu), np.array(nr), np.array(nx)