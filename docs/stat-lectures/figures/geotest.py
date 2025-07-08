
import cutil as cu


if __name__ == "__main__":
	import pylab as pl
	import scipy.stats as ss
	import numpy as np

	geom = ss.geom

	numargs = geom.numargs
	rv = geom( .2 )

	if False:

		fig = cu.get_fig_axes( 2, 1, 1, 1, 1, 1\
				, 6, 3\
				, .5, .5 )

		ax0 = fig.get_axes()[0]
		ax1 = fig.get_axes()[1]

		x = np.arange( 1, 31 )
		h1 = ax0.plot( x, rv.pmf( x ), 'o' )

		h2 = ax1.plot( x, rv.cdf( x ), 'o' )

		pl.show()

	# LGExample 3.26 Device Lifetimes

	r = .2
	s = .5
	al = .3

	g1 = geom( r )
	g2 = geom( s )

	mine = al*(2-r)/r**2. + (1.-al)*(2-s)/s**2. - (al/r + (1.-al)/s)**2.
	text = al*(1+r)/r**2. + (1.-al)*(1+s)/s**2. - (al/r + (1.-al)/s)**2.

	print al/r + (1.-al)/s

	print mine
	print text

	N = 1e6

	d1 = g1.rvs(N)
	d2 = g2.rvs(N)

	print d1.mean(), g1.stats('m'), d1.var(), g1.stats('v')
	print d2.mean(), g2.stats('m'), d2.var(), g2.stats('v')

	a = ss.bernoulli(al).rvs(N)
	nota = np.logical_not(a)

	d = (a*d1 + nota*d2)

	print d.mean(), d.var()




