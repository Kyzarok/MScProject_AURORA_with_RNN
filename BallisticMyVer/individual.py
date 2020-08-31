import math
import numpy as np

M = 1.0
NB_STEP = 50
DT = 1.0
FMAX = 200.0

class indiv():

  def __init__(self):
    self.bd = []

    self._gt = []
    self.cart_traj = []
    self.polar_traj = []
    
    self.theta = 0.0
    self.F = 0.0

    self.key = []

    self._entropy = 0.0

    self.novelty = 0.0

    self.max_point = []

    self.curiosity_score = 0.0


    self.key = 0.001
    self.mult = int(1/self.key)
  
  def set_bd(self, bd):
    self.bd = bd

  def get_bd(self):
    return self.bd

  def set_novelty(self, nov):
    self.novelty = nov
  
  def get_novelty(self):
    return self.novelty

  def get_max_point(self):
    return self.max_point

  def get_curiosity(self):
    return self.curiosity_score

  def increase_curiosity(self):
    self.curiosity_score += 1.0

  def decrease_curiosity(self):
    self.curiosity_score -= 0.5

  def is_dead(self):
    zero = np.zeros(2, 50)
    if zero == self.cart_traj:
      return True
    else:
      return False

  def get_key(self):
    return self.key

  def get_traj(self):
    trans = np.array(self.cart_traj)
    return trans.T

  def get_lstm_embed_traj(self):
    a_mod = (self.mult * self.mult) + 1
    b_mod = self.mult
    ret = np.zeros((1, 50))
    for i in range(NB_STEP):
      dim_0 = self.scaled_traj_image[0][i]
      dim_1 = self.scaled_traj_image[0][i + NB_STEP]
      a = round(dim_0 * self.mult) / self.mult
      b = round(dim_1 * self.mult) / self.mult
      a = a * a_mod
      b = b * b_mod
      ret[0][i] = a + b

    return ret

  def get_scaled_image(self, _max, _min):
    data = self.cart_traj
    den = (_max - _min).T
    for d in range(len(data)):
      for m in range(len(_min)):
        data[d][m] -= _min[m]
    data /= den
    res = [ r*2 - 1 for r in data]

    self.scaled_traj_image = self.get_traj_image(res)
    

    return self.scaled_traj_image

  def get_traj_image(self, data):
    ret = np.zeros((1, 100))
    for c in range(NB_STEP):
      ret[0, c] = data[c][0]
      ret[0, c + NB_STEP] = data[c][1]
    return ret

  def get_gt(self):
    return self._gt

  def get_entropy(self):
    return self._entropy

  def get_flat_traj(self, data):
    bound = len(self.cart_traj)
    for t in range(bound):
      data[0][t] = self.cart_traj[t][0]
      data[0][t+bound] = self.cart_traj[t][1]
    return data

  def get_flat_obs_size(self):
    return len(self.cart_traj) * len(self.cart_traj[0])

  def eval(self, ind):
    self.key = ind
    self.theta = ind[0] * math.pi/2
    self.F = ind[1]*FMAX
    self.simulate(self.F,self.theta)
    self._gt = self.desc_hardcoded()

  def simulate(self, F, theta):
    a = [F * math.cos(theta) / M, (F * math.sin(theta)-9.81) / M]
    if F * math.sin(theta) <= 9.81*3 :
      p = [0.0, 0.0]
      for t in range(NB_STEP):
        self.cart_traj.append(p)
        self.polar_traj.append(p)
    v = [0.0, 0.0] # Velocity
    p = [0.0, 0.0] # Position, second value is height
    polar = [0.0, 0.0]
    append_type = p.copy()
    self.cart_traj.append(append_type)
    self.polar_traj.append(polar)

    for t in range(NB_STEP - 1):
      v[0] += a[0] * DT
      v[1] += a[1] * DT
      p[0] += v[0] * DT
      p[1] += v[1] * DT
      a = [0.0, -9.81]
        
      if p[1] <= 0.0: # If puck has made contact with the ground
        p[1] = 0.0
        a[1] = -0.6*v[1] # Dumping Factor
        v[1] = 0.0
      polar = [np.linalg.norm(p), math.atan2(p[1], p[0])]
      append_type = p.copy()
      self.cart_traj.append(append_type)
      self.polar_traj.append(polar)

  def desc_fulldata(self):
    data = np.zeros(1, self.get_flat_obs_size())
    data = self.get_flat_traj(data)
    res = []
    for i in range(self.get_flat_obs_size()):
      res.append(data[0, i])
    return res

  def desc_genotype(self):
    res = [self.theta/(math.pi/2.0)*2-1, self.F/200.0*2-1]
    return res
  
  def desc_hardcoded(self):
    Vx = math.cos(self.theta) * self.F
    Vy = math.sin(self.theta) * self.F - 9.81
    Px = Vx/2.0
    Py = Vy/2.0
    tmax = (math.sin(self.theta) * self.F)/9.81 - 1
    res = [(Vx * tmax + Px )/2000*2-1, (-9.81* 0.5 * tmax*tmax + Vy * tmax + Py) / 2000 * 2 - 1] # quick normalization

    return res


