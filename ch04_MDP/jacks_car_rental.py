from common import MDP
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

POISSON_UB = 11

class JacksCarRental(MDP):
	def __init__(self, max_cars=10, max_move=5, move_cost=2, 
				 rental_reward=10, rental_rate=[3,4], return_rate=[3,2],
				 gamma=0.9, eps=1e-4, verbose=False):
		self.max_cars 		= max_cars
		self.max_move 		= max_move
		self.move_cost 		= move_cost
		self.rental_reward 	= rental_reward
		self.rental_rate	= rental_rate
		self.return_rate	= return_rate
		self.gamma			= gamma
		self.poisson_cache	= {}

		states_dim 			= [max_cars+1, max_cars+1]
		actions 			= list(range(-max_move, max_move+1))
		super().__init__(states_dim, actions, eps, verbose)


	def poisson(self, n, lam):
		key = (n, lam)
		if key not in self.poisson_cache.keys():
			self.poisson_cache[key] = np.exp(-lam) * lam**n / (np.math.factorial(n))

		return self.poisson_cache[key]

	def expected_returns(self, state, action):
		returns = - self.move_cost * abs(action)
		all_prob = 0
		for rental_b1 in range(0, POISSON_UB):
			for rental_b2 in range(0, POISSON_UB):
				num_cars_b1 = min(state[0] - action, self.max_cars)
				num_cars_b2 = min(state[1] + action, self.max_cars)

				rented_cars_b1 = min(num_cars_b1, rental_b1)
				rented_cars_b2 = min(num_cars_b2, rental_b2)
				num_cars_b1 -= rented_cars_b1
				num_cars_b2 -= rented_cars_b2

				reward = (rented_cars_b1 + rented_cars_b2) * self.rental_reward
				
				# prob = self.poisson(rental_b1, self.rental_rate[0]) * \
				# 	   self.poisson(rental_b2, self.rental_rate[1])  
				

				for return_b1 in range(0, POISSON_UB):
					for return_b2 in range(0, POISSON_UB):
						num_cars_b1 = min(num_cars_b1 + return_b1, self.max_cars)
						num_cars_b2 = min(num_cars_b2 + return_b2, self.max_cars)
						next_state = (num_cars_b1, num_cars_b2)
						prob =  self.poisson(rental_b1, self.rental_rate[0]) * \
							    self.poisson(rental_b2, self.rental_rate[1]) * \
							    self.poisson(return_b1, self.return_rate[0]) * \
					   			self.poisson(return_b2, self.return_rate[1])  
						# print(reward)
						returns += prob * (reward + self.gamma * self.value_state[next_state])
						# print(returns)

		return returns

	def plot_value_function(self):
		fig = plt.figure(figsize=(12,9))
		ax = fig.gca(projection='3d')

		x, y = list(zip(*self.states))
		vals = []

		for i in range(self.max_cars+1):
			for j in range(self.max_cars+1):
				vals.append(self.value_state[(x[i], y[j])])

		surf = ax.scatter(x, y, vals)
		ax.set_xlabel("# of Cars in branch 1", size=18)
		ax.set_xlim(0, 10)
		ax.set_ylabel("# of Cars in Branch 2", size=18)
		ax.set_ylim(0, 10)
		ax.set_zlabel("Expected Profit", size=18)
		ax.set_zlim(min(vals), max(vals))
		plt.show()

	def plot_policy(self):
		fig_3d = plt.figure(figsize=(12,9))
		ax_3d = fig_3d.gca(projection='3d')

		x, y = list(zip(*self.states))
		vals = []

		for i in range(self.max_cars+1):
			for j in range(self.max_cars+1):
				vals.append(self.policy[(x[i], y[j])])

		surf = ax_3d.scatter(x, y, vals)
		ax_3d.view_init(30, 0)
		ax_3d.set_xlabel("# of Cars in branch 1", size=18)
		ax_3d.set_xlim(0, 10)
		ax_3d.set_ylabel("# of Cars in Branch 2", size=18)
		ax_3d.set_ylim(0, 10)
		ax_3d.set_zlabel("# of Cars Moved at Night", size=18)
		ax_3d.set_zlim(-self.max_move, self.max_move)

		# plt.contour(x, y, np.array(vals).reshape((self.max_cars+1, self.max_cars+1)))
		plt.show()