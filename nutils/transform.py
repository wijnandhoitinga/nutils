from __future__ import division
from . import cache, numeric, util, _
import fractions


def mask2nums( mask ):
  nums = []
  i = 0
  while mask:
    if mask & 1:
      nums.append( i )
    mask >>= 1
    i += 1
  return nums


class Factors( object ):

  def __init__( self ):
    self.primes = []
    self.D = [ None, None, 0 ]
    self.q = 1

  def _next( self ):
    self.q += 1
    if not self.D[self.q]:
      iprime = len(self.primes)
      self.primes.append( self.q )
      self.D[self.q] = 1 << iprime
    nums = mask2nums( self.D[self.q] )
    self.D += [0] * ( self.q + self.primes[nums[-1]] - len(self.D) + 1 )
    for n in nums:
      self.D[self.primes[n]+self.q] |= 1 << n

  def mask2primes( self, mask ):
    primes = []
    for prime in self.primes:
      if mask & 1:
        primes.append( prime )
      mask >>= 1
      if not mask:
        return primes
    raise Exception, 'not enough primes for given mask'

  def getmask( self, q ):
    while self.q < q:
      self._next()
    return self.D[q]

  def common( self, numbers ):
    numbers = sorted( map( abs, numbers ) )
    if numbers[-1] == 1:
      return 1
    while numbers[0] == 0:
      numbers = numbers[1:]
    if numbers[0] == 1:
      return 1
    mask = self.getmask( numbers[0] )
    for q in numbers[1:]:
      mask &= self.getmask(q)
      if not mask:
        return 1
    return util.product( self.mask2primes(mask) )
      
  def __getitem__( self, q ):
    assert isinstance( q, int ) and q > 1
    return self.mask2primes( self.getmask(q) )
    

factors = Factors()

#print '2', factors[2]
#print '3', factors[3]
#print '4', factors[4]
#print '5', factors[5]
#print '6', factors[6]
#print '7', factors[7]
#print '8', factors[8]


def Root( ndims ):
  return Identity( ndims )


def Identity( ndims ):
  return AffineTrans( numeric.array(1), numeric.zeros(ndims,dtype=int), 1, 0 )


def Point( sign ):
  return AffineTrans( numeric.zeros([1,0],dtype=int), numeric.zeros(1,dtype=int), 1, sign )


def ScaleUniform( ndims, invfactor ):
  assert isinstance( invfactor, int )
  return AffineTrans( numeric.array(1), numeric.zeros(ndims,dtype=int), invfactor, 0 )


def Scale( factors ):
  prec = 2310 # 2 * 3 * 5 * 7 * 11
  f = numeric.round( numeric.asarray(factors) * prec )
  return affinetrans( f, numeric.zeros(len(f),dtype=int), prec, 0 )


def affinetrans( linear, shift, numer, sign ):
  assert numer > 0
  nums = numeric.empty( linear.size + shift.size + 1, dtype=int )
  nums[:linear.size] = linear.flat
  nums[linear.size:-1] = shift
  nums[-1] = numer
  common = factors.common( nums )
  while common != 1:
    nums //= common
    common = factors.common( nums )
  return AffineTrans( nums[:linear.size].reshape(linear.shape), nums[linear.size:-1], nums[-1], sign )
  

class AffineTrans( cache.Immutable ):
  __slots__ = 'linear', 'shift', 'numer', 'sign', 'todim', 'fromdim'

  def __init__( self, linear, shift, numer, sign ):
    if linear.ndim == 0:
      self.todim, = self.fromdim, = shift.shape
    elif linear.ndim == 1:
      self.todim, = self.fromdim, = linear.shape
      assert shift.shape == (self.todim,)
    elif linear.ndim == 2:
      self.todim, self.fromdim = linear.shape
      assert shift.shape == (self.todim,)
    else:
      raise Exception, 'invalid linear operator with shape %s' % (linear.shape,)
    assert isinstance( numer, int ) and numer >= 1
    assert self.todim == self.fromdim and sign == 0 or self.todim == self.fromdim + 1 and sign in (-1,1)
    assert linear.dtype == int
    assert shift.dtype == int
    self.linear = linear
    self.shift = shift
    self.numer = numer
    self.sign = sign

  def apply( self, points, axis=-1 ):
    assert axis==-1 # for now
    assert points.shape[axis] == self.fromdim
    return ( numeric.dot( points, self.linear.T, axis ) + self.shift if self.linear.ndim == 2
                              else points * self.linear + self.shift ) / float(self.numer)

  @property
  def matrix( self ):
    scaled = self.linear / float(self.numer)
    return scaled if scaled.ndim == 2 \
      else numeric.diag( scaled ) if scaled.ndim == 1 \
      else numeric.eye( self.fromdim ) * scaled

  @property
  def offset( self ):
    return self.shift / float(self.numer)

  @property
  def invmatrix( self ):
    return numeric.inv( self.matrix )

  @property
  def intdet( self ): # divide by numer to get det
    return numeric.exterior( self.linear ) * self.sign if self.fromdim != self.todim \
      else numeric.det( self.linear ) if self.linear.ndim == 2 \
      else numeric.product( self.linear ) if self.linear.ndim == 1 \
      else self.linear**self.todim

  @property
  def det( self ):
    return self.intdet / float(self.numer)

  @property
  def squaremat( self ):
    return self.linear if self.fromdim == self.todim \
      else numeric.concatenate( [ self.linear, self.intdet[:,_] ], axis=1 )

  @property
  def invsquaremat( self ):
    if self.todim == 2:
      (a11,a12),(a21,a22) = self.squaremat
      A = numeric.array( ((a22,-a12),(-a21,a11)) ) * self.numer
      numer = a11 * a22 - a12 * a21
    else:
      raise NotImplementedError
    if numer < 0:
      numer = -numer
      A = -A
    return A, numer
    
  def __add__( self, shift ):
    scaled = numeric.asarray( shift ) * self.numer
    scaledint = numeric.round( scaled )
    assert numeric.equal( scaled, scaledint ).all()
    return affinetrans( self.linear, self.shift + scaledint, self.numer, self.sign )

  def __mul__( self, other ):
    linear = numeric.dot( self.linear, other.linear ) if self.linear.ndim == other.linear.ndim == 2 \
        else ( self.linear.T * other.linear.T ).T
    shift = ( numeric.dot( self.linear, other.shift ) if self.linear.ndim == 2
          else self.linear * other.shift ) + other.numer * self.shift
    numer = self.numer * other.numer
    return affinetrans( linear, shift, numer, self.sign or other.sign )

  def __str__( self ):
    return ( '%s+%sx/%s' % ( self.shift.tolist(), self.linear.tolist(), self.numer ) ).replace( ' ', '' )

  def __repr__( self ):
    return 'AffineTrans(%s)' % self
    

## UTILITY FUNCTIONS


def tensor( trans1, trans2 ):
  fromdim = trans1.fromdim + trans2.fromdim
  todim = trans1.todim + trans2.todim

  numer = trans1.numer * trans2.numer

  shift = numeric.zeros( todim, dtype=int )
  shift[:trans1.todim] = trans1.shift * trans2.numer
  shift[trans1.todim:] = trans2.shift * trans1.numer

  matrix = numeric.zeros( [todim,fromdim], dtype=int )
  matrix[:trans1.todim,:trans1.fromdim] = trans1.matrix * trans2.numer
  matrix[trans1.todim:,trans1.fromdim:] = trans2.matrix * trans1.numer

  # TODO generalize
  if trans1.todim == 1 and trans1.fromdim == 0 and trans2.todim == trans2.fromdim == 1:
    sign = -trans1.sign
  elif trans1.todim == trans1.fromdim == 1 and trans2.todim == 1 and trans2.fromdim == 0:
    sign = trans2.sign
  elif trans1.todim == 1 and trans1.fromdim == 0 and trans2.todim == trans2.fromdim == 2:
    sign = trans1.sign
  elif trans1.todim == trans1.fromdim == 1 and trans2.todim == 2 and trans2.fromdim == 1:
    sign = trans2.sign
  else:
    raise NotImplementedError

  return affinetrans( matrix, shift, numer, sign )

def split_linear_offset( trans ):
  if isinstance( trans, Affine ):
    return trans.transform, trans.shift
  return trans, numeric.zeros( trans.todim, dtype=int )

def canonical( trans ):
  # keep at lowest ndims possible
  for i in reversed( range(1,len(trans)) ):
    if trans[i].fromdim == trans[0].todim:
      break
    while i < len(trans):
      scale, updim = trans[i-1:i+1]
      if updim.todim != updim.fromdim + 1 or scale.linear.ndim != 0:
        break
      Ainv, Ainv_numer = updim.invsquaremat
      x = numeric.dot( Ainv, scale.shift * updim.numer + scale.linear * updim.shift )
      offset_scale = x[:-1]
      offset_updim = x[-1] * updim.intdet
      newscale = affinetrans( scale.linear, offset_scale, scale.numer * Ainv_numer, scale.sign )
      newupdim = affinetrans( updim.linear, offset_updim, updim.numer * Ainv_numer, updim.sign )
      if newupdim != updim:
        break
      assert newupdim * newscale == scale * updim
      trans = trans[:i-1] + (newupdim,newscale) + trans[i+1:]
      i += 1
  return trans

def prioritize( trans, ndims ):
  # assuming canonical, move up to ndims asap, stay at ndims alap
  for i in range(1,len(trans)):
    while i and trans[i].todim < ndims:
      updim, scale = trans[i-1:i+1]
      if updim.todim != updim.fromdim + 1 or scale.linear.ndim != 0:
        break
      c = numeric.dot( updim.linear, scale.offset ) + updim.offset * scale.numer - scale.linear * updim.offset
      newscale = affinetrans( scale.linear * updim.numer, c, updim.numer * scale.numer, scale.sign )
      assert newscale * updim == updim * scale
      trans = trans[:i-1] + (newscale,updim) + trans[i+1:]
      i -= 1
  return trans

def solve( target, trans ):
  # find trans * result == target, assuming possible
  fromdim = target.fromdim
  todim = target.todim
  assert todim == fromdim + 1
  assert fromdim == trans.fromdim
  assert todim == trans.todim
  E, e = split_linear_offset( target )
  D, d = split_linear_offset( trans )
  Dmat = D.matrix
  DD = numeric.dot( Dmat.T, Dmat )
  print 'DD', DD
  DDD = numeric.solve( DD, Dmat.T )
  print 'DDD', DDD
  C = numeric.dot( DDD, E.matrix )
  c = numeric.dot( DDD, e-d )
  result = Linear( C ) + c
  print '1.', trans, ',', result
  print '2.', target
  assert trans * result == target
  return result
