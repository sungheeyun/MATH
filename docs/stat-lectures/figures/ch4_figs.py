
if __name__ == "__main__":
	import cutil as cu
	import stutil as su

	import scipy.stats as ss
	import numpy as np

	ALL = True

	items = [ 'cdfpdf1' ]
	items = [ 'unif_ff' ]
	items = [ 'fig4_4' ]


	import pylab as pl

	if ALL or 'cdfpdf1' in items:
		rv = ss.binom( 3, .5 )


		#fig = su.get_fa(2,1)
		#ax1 = fig.get_axes()[0]
		#ax2 = fig.get_axes()[1]

		fig = pl.figure()
		ax1 = fig.add_subplot(211)
		ax2 = fig.add_subplot(212)


		xs = np.linspace(-3,6,1000)
		su.plotcdf( ax1, xs, rv.cdf(xs) )
		ax1.set_title( r'$F_X(x)$' )

		ks = np.arange(4)
		su.plotpmf( ax2, ks, rv.pmf(ks) )
		ax2.set_title( r'$f_X(x)$' )

		ax1.set_xlim( (-1, 4 ))
		ax2.set_xlim( (-1, 4 ))

		su.savefigs( fig, 'cdfpdf1' )

	if ALL or 'unif_ff' in items:
		rv = ss.uniform( 0., 1. )

		fig = pl.figure()
		ax1 = fig.add_subplot(211)
		ax2 = fig.add_subplot(212)

		xs = np.linspace(-1,2,1000)
		su.plotcdf( ax1, xs, rv.cdf(xs) )
		ax1.set_title( r'$F_X(x)$' )

		su.plotpdf( ax2, xs, rv.pdf(xs) )
		ax2.set_title( r'$f_X(x)$' )

		su.savefigs( fig, 'unif_ff' )

	if ALL or 'fig4_4' in items:
		mu = 1.
		sig = 1.
		rv = ss.norm( mu, sig )

		fig = pl.figure()
		ax1 = fig.add_subplot(121)
		ax2 = fig.add_subplot(122)

		xmin = mu-3.*sig
		xmax = mu+3.*sig
		xs = np.linspace(xmin,xmax,1000)

		a1, b1 = 2, 2.2
		a2, b2 = 1.5, 2.5

		color = (.5,.5,.5)
		def color_area( ax, a, b, s1, s2 ):
			xs1 = np.linspace(a,b,1000)
			ys1 = rv.pdf(xs1)
			xs1 = np.r_[xs1[0],xs1,xs1[-1]]
			ys1 = np.r_[0.,ys1,0.]
			ax.plot( xs, rv.pdf(xs) )
			ax.plot( [xs.min(), xs.max()], [0,0], 'k' )
			ax.fill( xs1, ys1, color = color, ec = color )
			ax.text( a, -.01, s1, ha = 'right', va = 'center' )
			ax.text( b, -.01, s2, ha = 'left', va = 'center' )
			ax.set_ylim( (-.02,.45) )

		color_area( ax1, a1, b1, '$x$', '$x+dx$' )
		color_area( ax2, a2, b2, '$a$', '$b$' )

		

		su.savefigs( fig, 'fig4_4' )


	pl.show()

