import gym
import torch
from torch import nn
from torch import optim
import numpy as np
from tensorboardX import SummaryWriter

BATCH_SIZE = 100
HIDDEN_SIZE = 128
PERCENTILE = 70
LEARNING_RATE = 0.001
GAMMA = 0.9
TARGET_MEAN_REWARD = 0.49

class onehotencoder(gym.ObservationWrapper):
	def __init__(self, env):
		super(onehotencoder, self).__init__(env)
		self.observation_space = gym.spaces.Box(0.0, 1.0, (env.observation_space.n,))
	def observation(self, observation):
		res = np.copy(self.observation_space.low)
		res[observation] = 1
		return res

env = onehotencoder(gym.make("FrozenLake-v0"))
# env = gym.wrappers.Monitor(env, directory="rec", force=True)
s_size = env.observation_space.shape[0]
a_size = env.action_space.n
hidden_size = HIDDEN_SIZE

writer = SummaryWriter()

class CEpi(nn.Module):
	def __init__(self, hidden_size, s_size, a_size):
		super(CEpi, self).__init__()
		self.net = nn.Sequential(
			nn.Linear(s_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, a_size)
			)
	def forward(self, s):
		return self.net(s)

net = CEpi(hidden_size, s_size, a_size)
sm = nn.Softmax(dim=1)

def batcher(env):
	s = env.reset()

	iter_no = 0

	batch = []
	buffer = {"s":[],"a":[],"r":[]}
	buffer["G"] = 0

	while True:

		aP = sm(net(torch.FloatTensor([s]))).data.numpy()[0]
		a = np.random.choice(len(aP), p = aP)
		n_s, r, term, _ = env.step(a)

		buffer["s"].append(s)
		buffer["a"].append(a)
		buffer["r"].append(r)

		buffer["G"] += r

		s = n_s

		if term:
			batch.append(buffer)
			s = env.reset()
			iter_no += 1
			writer.add_scalar("Episode Return",buffer["G"], iter_no)
			buffer = {"s":[],"a":[],"r":[]}
			buffer["G"] = 0
			if len(batch) == BATCH_SIZE:
				yield batch
				batch = []

def filter_batches(batch, percentile = PERCENTILE):
	Gs = [sum([r*(GAMMA**n) for n,r in enumerate(buffer["r"])]) for buffer in batch]
	print(Gs)
	G_bound = np.percentile(Gs, PERCENTILE)
	G_mean = float(np.mean(Gs))

	ss = []
	a_s = []
	elite_batch = []
	for n,buffer in enumerate(batch):
		if Gs[n] >= G_bound:
			if Gs[n] == 0:
				continue
			ss.extend([s for s in buffer["s"]])
			a_s.extend([a for a in buffer["a"]])
			elite_batch.append(buffer)
	return elite_batch, torch.FloatTensor(ss), torch.LongTensor(a_s), G_bound, G_mean


optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

full_batch = []
for it_no, batch in enumerate(batcher(env)):
	full_batch, ss, a_s, G_bound, G_mean = filter_batches(full_batch+batch)
	if not full_batch:
		continue
	full_batch = full_batch[-500:]
	optimizer.zero_grad()
	aPs = net(torch.FloatTensor(ss))
	loss_v = nn.CrossEntropyLoss()(aPs, torch.LongTensor(a_s))
	loss_v.backward()
	optimizer.step()
	print(it_no,"Loss: ", loss_v.item(), "G_mean", G_mean,"G_bound", G_bound)
	writer.add_scalar("Loss",loss_v.item(),it_no)
	writer.add_scalar("G_bound",G_bound,it_no)
	writer.add_scalar("G_mean",G_mean,it_no)
	if G_mean > TARGET_MEAN_REWARD:
		print("Solved!", it_no)
		break
writer.close()
