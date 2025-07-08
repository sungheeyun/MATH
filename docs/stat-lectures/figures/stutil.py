
import cutil as cu


def plotcdf( ax, xs, cdf, *pargs, **kargs ):
	ax.plot( xs, cdf, *pargs, **kargs )
	ax.set_ylim((-.1,1.1))


def plotpdf( ax, xs, cdf, *pargs, **kargs ):
	r = .1
	ax.plot( xs, cdf, *pargs, **kargs )
	#ylim = ax.get_ylim()
	#ymax = ylim[1]
	ymax = cdf.max()
	ax.set_ylim( ( -r*ymax, (1+r)*ymax ) )


def plotpmf( ax, xs, pmfs, arrow = True ):
	import pylab as pl
	r = .1

	if arrow:
		ax.plot( xs, pmfs, 'b^' )
		ax.vlines( xs, [0], pmfs, color = 'b' )
	else:
		markerline, stemlines, baseline = ax.stem( xs, pmfs )
		#pl.setp( baseline, visible = False )
		pl.setp( baseline, 'visible', False )

	xmin, xmax = xs.min(), xs.max()
	#ax.set_xlim( ( (1+r)*xmin-r*xmax, (1+r)*xmax-r*xmin ) )
	ax.set_xlim( ( xmin-.5, xmax+.5 ) )
	ax.set_ylim( ( 0.0, (1+r)*pmfs.max() ) )

def get_fa( m, n ):
	return cu.get_fig_axes( m, n\
			, 1, 1, .7, 1\
			, 6, 3\
			, .5, 1 )

def draw_pmf( ax, k, pmf ):
	plotpmf( ax, k, pmf )
	ax.set_xlabel( '$k$' )
	ax.set_ylabel( '$p_X(k)$' )

def savefigs( fig, name ):
	Exts = [ 'eps', 'pdf' ]
	for ext in Exts:
		fig.savefig( '%s.%s' % ( name, ext ), format = ext )


