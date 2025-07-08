
dolist = [ 'pr2.108']
dolist = [ 'pr2.76']
dolist = [ 'pr3.57']
dolist = [ 'pr3.66']
dolist = [ 'pr3.67']
dolist = [ 'skewness' ]

def savefigs( fig, name ):
	exts = [ 'png', 'eps' ]
	exts = [ 'pdf', 'eps' ]
	for ext in exts:
		fig.savefig( '%s.%s' % ( name, ext ), format = ext )

if 'pr2.76' in dolist:
	import pylab as pl
	import numpy as np

	K = 10

	a = np.linspace(50,50-K+1,K)
	b = np.linspace(100,100-K+1,K)
	c = a/b
	d = np.cumprod(c)

	k = np.linspace(1,10,10)

	print d

	fig = pl.figure()
	ax1 = fig.add_subplot(211)
	ax2 = fig.add_subplot(212)
	ax1.plot(k,d,'o-')
	ax1.set_xlabel('k')
	ax1.set_ylabel('probability')

	ax2.semilogy(k,d,'o-')
	ax2.set_xlabel('k')
	ax2.set_ylabel('probability (log-scale)')

	fig.savefig( 'pr02_76.png', format = 'png' )
	fig.savefig( 'pr02_76.eps', format = 'eps' )

	pl.show()

if 'pr2.108' in dolist:
	from numpy import matrix
	import numpy.linalg as la

	# the transition matrix for the Markov chains
	A = matrix([[2./3,1./6],[1./3,5./6]])

	# initial probability distribution
	p1 = matrix([[.5],[.5]])

	p2 = A* p1
	p3 = A* p2
	p4 = A*p3

	print p2,p3,p4

	# eigenvalue decomposition of A
	L, V = la.eig( A ) # L: eigenvalues, V: eigenvectors
	Vi = la.inv( V ) # the inverse of V

	# examine the eigenvalues of A
	print L

	# Since the first eigenvalue is .5, and the second is 1,
	# at the equilibrium, the probabilities converge to
	# those corresponding to 1.
	# Then the equilibrium can be calculated as
	print V[:,1] * Vi[1,:] * p1

	
if 'pr3.57' in dolist:
	import scipy.stats as ss
	x = ss.poisson(1)
	y = ss.poisson(2)
	z = ss.poisson(4)

	print 'Pr{X=0} = %g' % x.pmf(0)
	print 'Pr{Y<=2} = %g' % y.cdf(2)
	print 'Pr{X=0}*Pr{Y<=2} = %g' % (x.pmf(0)*y.cdf(2))
	print 'Pr{X=0}*Pr{Y=0}*Pr{Z>5} = %g' % (x.pmf(0)*y.cdf(0)*z.sf(5))


if 'pr3.66' in dolist:
	print 'LGProb 3.66'

	import scipy.stats as ss
	n = 10000
	p = 1e-3
	x = ss.poisson( n * p )
	print 'Pr{X=0} = %g' % x.pmf(0)
	y = ss.poisson( 2. * n * p )
	print 'Pr{Y<10} = %g' % y.cdf(9)

	print 'Pr{X<%d} >= 0.99' % x.ppf( .99 )


if 'pr3.67' in dolist:
	print 'LGProb 3.67'

	import scipy.stats as ss
	n = 10000
	p = 1e-6
	al = n * p
	x = ss.poisson( al )
	y = ss.binom(n,p)

	print 'Pr{N=0} =', x.cdf(0)
	print 'Pr{N<=3} =', x.cdf(3)

if 'skewness' in dolist:
	import numpy as np
	import pylab as pl
	import scipy.stats as ss
	import stutil as su
	import cutil as cu

	# binomial random variable
	n = 10

	ps = [.3, .5, .8]
	num = len(ps)
	ks = np.arange(n+1)

	brvs = [ss.binom(n,p) for p in ps]

	fig = cu.get_fig_axes( num, 1\
			, 1., 1., .7, .7\
			, 4., 2.\
			, 0., .7 )
	for idx, rv in enumerate( brvs ):
		ax = fig.get_axes()[idx]
		#ax.set_xlabel( r'$n=%g,\ p=%g$' % ( n, ps[idx] ) )
		ax.set_xlabel( r'$n=%g,\ p=%g$' % rv.args )
		su.plotpmf( ax, ks, rv.pmf(ks) )

	savefigs( fig, 'binoms' )

	p = np.linspace(.01,.99,10000)
	s = (1.-2.*p)/np.sqrt(n*p*(1.-p))

	fig = pl.figure()
	ax = fig.add_subplot(111)
	ax.plot(p,s)
	ax.set_xlabel( '$p$' )
	#ax.set_title( r'$\frac{1-2p}{\sqrt{np(1-p)}}$' )
	savefigs( fig, 'skew_binom' )

	# geometric random variable
	ps = [.2, .6, .95]
	num = len(ps)
	ks = np.arange(1,21)

	brvs = [ss.geom(p) for p in ps]
	fig = cu.get_fig_axes( num, 1\
			, 1., 1., .7, .7\
			, 4., 2.\
			, 0., .7 )
	for idx, rv in enumerate( brvs ):
		ax = fig.get_axes()[idx]
		#ax.set_xlabel( r'$n=%g,\ p=%g$' % ( n, ps[idx] ) )
		su.plotpmf( ax, ks, rv.pmf(ks) )
		ax.set_xlabel( r'$p=%g$' % rv.args )
		if True:
			xlim, ylim = ax.get_xlim(), ax.get_ylim()
			ax.plot( rv.moment(1) * np.ones(2), ylim, 'g-', lw=1.5 )
			ax.set_xlim(xlim)
			ax.set_ylim(ylim)

	savefigs( fig, 'geoms' )


	p = np.linspace(0,.95,10000)
	s = (2.-p)/np.sqrt(1.-p)

	fig = pl.figure()
	ax = fig.add_subplot(111)
	ax.plot(p,s)
	ax.set_xlabel( '$p$' )

	savefigs( fig, 'skew_geom' )
	
	# Poisson random variable
	als = [1., 5., 10.]
	num = len(als)
	ks = np.arange(1,21)

	brvs = [ss.poisson(al) for al in als]
	fig = cu.get_fig_axes( num, 1\
			, 1., 1., .7, .7\
			, 4., 2.\
			, 0., .7 )
	for idx, rv in enumerate( brvs ):
		ax = fig.get_axes()[idx]
		su.plotpmf( ax, ks, rv.pmf(ks) )
		ax.set_xlabel( r'$\alpha=%g$' % ( als[idx] ) )
		if False:
			xlim, ylim = ax.get_xlim(), ax.get_ylim()
			ax.plot( rv.moment(1) * np.ones(2), ylim, 'r-', lw=2. )
			ax.set_xlim(xlim)
			ax.set_ylim(ylim)

	savefigs( fig, 'posss' )


	pl.show()
