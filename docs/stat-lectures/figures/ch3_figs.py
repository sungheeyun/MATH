
import cutil as cu
import stutil as su


if __name__ == "__main__":
	import scipy.stats as ss
	import numpy as np

	rvs = [ 'bernoulli' ]
	rvs = [ 'geom' ]
	rvs = [ 'binom' ]
	rvs = [ 'poss' ]
	rvs = [ 'bernoulli', 'binom', 'geom', 'poss' ]
	FIG = True

	if FIG:
		import pylab as pl

	if 'bernoulli' in rvs:
		p = .6
		rv = ss.bernoulli( p )

		k = np.arange(2)
		pmf =  rv.pmf(k)

		if FIG:
			fig = pl.figure()
			ax = fig.add_subplot(111)
			su.draw_pmf( ax, k, pmf )
			ax.set_title( r'$p=%g$' % p )
			su.savefigs( fig, 'bernoulli_pmf' )

	if 'binom' in rvs:
		n1, p1 = 5, .3
		n2, p2 = 10, .6
		rv1 = ss.binom( n1, p1 )
		rv2 = ss.binom( n2, p2 )

		k1 = np.arange(n1+1)
		pmf1 =  rv1.pmf(k1)
		k2 = np.arange(n2+1)
		pmf2 =  rv2.pmf(k2)

		if FIG:
			fig = su.get_fa( 2, 1 )
			ax1, ax2 = fig.get_axes()
			su.draw_pmf( ax1, k1, pmf1 )
			su.draw_pmf( ax2, k2, pmf2 )
			ax1.set_title( r'$n=%g,\ p=%g$' % ( n1, p1 ) )
			ax2.set_title( r'$n=%g,\ p=%g$' % ( n2, p2 ) )
			su.savefigs( fig, 'binom_pmf' )

	if 'geom' in rvs:
		p1, p2 = .3, .5
		rv1, rv2 = ss.geom( p1 ), ss.geom( p2 )

		k = np.arange(1,20)
		pmf1 = rv1.pmf(k)
		pmf2 = rv2.pmf(k)

		if FIG:
			fig = su.get_fa( 2, 1 )
			#fig.suptitle( 'The geometric random variables', fontsize = 15 )
			ax1, ax2 = fig.get_axes()
			su.draw_pmf( ax1, k, pmf1 )
			su.draw_pmf( ax2, k, pmf2 )
			ax1.set_title( r'$p=%g$' % ( p1 ) )
			ax2.set_title( r'$p=%g$' % ( p2 ) )

			su.savefigs( fig, 'geom_pmf' )

	if 'poss' in rvs:
		als = [.75, 3., 9.]
		k = np.arange(25)
		rvs = [ ss.poisson( al ) for al in als ]
		pmfs = [ rv.pmf(k) for rv in rvs ]

		if FIG:
			fig = cu.get_fig_axes( len(als), 1\
					, 1, 1, .5, .5\
					, 6, 3\
					, .5, .8 )
			axs = fig.get_axes()

			for idx, al in enumerate(als):
				ax = axs[idx]
				su.draw_pmf( ax, k, pmfs[idx] )
				ax.set_title( r'$\alpha=%g$' % ( al ) )

			su.savefigs( fig, 'poss_pmf' )

	if FIG:
		pl.show()
