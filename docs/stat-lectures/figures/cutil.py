#!/user/svn/local/linux-Python-2.6.2/bin/python

import os, sys, time, copy, math
import re
import logging, subprocess

def dbmsg(ln,*msg):
	print('%s:%d:' % (filenamewopath(__file__),ln), end="")
	print(', '.join([str(x) for x in msg]))

def assertFatal(*msg):
	print('FATAL!', end="")
	print(', '.join([str(x) for x in msg]))
	assert False

def same_text_files(fn1,fn2):
	def get_lines(fn):
		fid = open(fn)
		l = []
		for line in fid:
			l.append(line)
		return l

	llist1 = get_lines(fn1)
	llist2 = get_lines(fn2)

	return llist1 == llist2

def is_monotone(x):
	assert len(x)>1, x
	return np.all(np.diff(np.array(x))>=0) or np.all(np.diff(np.array(x))<=0)

def is_nondec(x):
	assert len(x) > 1, x
	return np.all(np.diff(np.array(x))>=0)

def row_col_nums4subplot(n):
	ncol = math.floor(math.sqrt(n))
	nrow = math.ceil(n/ncol)
	assert nrow.is_integer() and ncol.is_integer(), (ncol,nrow)
	return int(nrow), int(ncol)

def last_mono_start_idx(x):
	assert len(x) > 1, len(x)

	xmin, xmax = np.min(x), np.max(x)
	eps = (xmax-xmin)*1e-3

	xdiff = np.diff(x)

	sidx = 0

	state = None
	for i in range(len(xdiff)):
		if xdiff[i] > eps:
			cur = 'inc'
		elif xdiff[i] < -eps:
			cur = 'dec'
		else:
			cur = None

		if state is None and cur == 'inc':
			state = 'inc'
		elif state is None and cur == 'dec':
			state = 'dec'
		elif state == 'dec' and cur == 'inc':
			sidx = i
			state = 'inc'
		elif state == 'inc' and cur == 'dec':
			sidx = i
			state = 'dec'

	return sidx


def in_prod(a,b):
	assert len(a) == len(b)

	return sum([a[i]*b[i] for i in range(len(a))])

def norm(vals):
	return math.sqrt(sum([val ** 2. for val in vals]))

def rms(y):
	assert len(y) > 0, len(y)
	return math.sqrt(sum([e ** 2.0 for e in y])/len(y))

def int2strfill(num,n):
	assert type(num) is int, (num)
	return str(num).zfill(n)

def now2str():
	return time.strftime('%Y_%m_%d_%H_%M_%S_%a')

def dn2str(dn):
	return fn2str(dn)

logger_name_set = set()

def getLogger(name,filenames=None,level=logging.DEBUG):
	if name in logger_name_set:
		return logging.getLogger(name)

	logger_name_set.add(name)

	# creat logger
	logger = logging.getLogger(name)
	logger.setLevel(level)

	if not filenames is None:
		if not isvec(filenames):
			filenames = [filenames]

		fids = []
		for filename in filenames:
			fids.append(open('%s.log'%filename,'w'))

	else:
		fids = [None]


	for fid in fids:
		# add ch to logger
		ch = getStreamHandler(fid,level)
		logger.addHandler(ch)

	return logger


def getStreamHandler(fid=None,level=logging.DEBUG):
	# create console handler and set level to debug
	ch = logging.StreamHandler(strm=fid)
	ch.setLevel(level)

	# create formatter
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'
		#,datefmt = '%a, %d %b %Y %H:%M:%S'\
		,datefmt = '%a, %d %b %H:%M:%S'\
		)

	# add formatter to ch
	ch.setFormatter(formatter)

	return ch

def str2str_num_dict(s):
	s = s.strip()
	assert s[0] == '{' and s[-1] == '}', s
	s = s[1:-1]

	res = {}
	for pair in s.split(','):
		key, val = [x.strip() for x in pair.split(':')]
		assert len(key) >= 3 and key[0] == "'" and key[-1] == "'", (key)
		res[key[1:-1]] = float(val)

	return res

def getanswer(s):
	sys.stdout.write(s)
	return sys.stdin.readline()[:-1]

def popen(cmmd):
	p = subprocess.Popen( cmmd, shell=True,\
		stdin=subprocess.PIPE,\
		stdout=subprocess.PIPE,\
		stderr=subprocess.STDOUT,\
		close_fds=True)

	return p.stdout

def ordinal(d):
	assert type(d) is int and d >= 0, d

	if d % 10 == 1 and not d % 100 == 11:
		return '%dst' % d

	if d % 10 == 2 and not d % 100 == 12:
		return '%dnd' % d

	if d % 10 == 3 and not d % 100 == 13:
		return '%drd' % d

	return '%dth' % d


def cmp8list(x,y,l):
	def _longest_match_index_(a):
		res = (-1,'')
		for i, b in enumerate(l):
			if a.startswith(b):
				if len(b) > len(res[1]):
					res = (i,b)
		assert res[0] >= 0, (i,a,l)
		return res[0]

	ix = _longest_match_index_(x)
	iy = _longest_match_index_(y)

	return ix - iy

def stop():
	assert False, 'STOP!'

def tic():
	global tictoct0
	tictoct0 = time.time()

def toc( s, file = None ):
	elp = time.time()-tictoct0
	if elp > .1:
		if file:
			postfix = ' (%s)' % file
		else:
			postfix = ''

		print('%s took %g sec.%s' % ( s, elp, postfix ))

def issubset(a,b):
	return set(a).issubset(b)

def issameset(a,b):
	return set(a) == set(b)

def isdisjoint(a,b):
	return set(a).isdisjoint(b)

def islogical(x):
	return x in [True,False]

def isodd(x):
	return not iseven(x)

def iseven(x):
	assert type(x) is int
	return x % 2 == 0

def isnumeric(x):
	return type(x) is int or type(x) is float

def mustnumeric(x):
	assert isnumeric(x), x

def exists(fn):
	return os.path.exists(fn)

def isfile(fn):
	return os.path.isfile(fn)

def isdir(fn):
	return os.path.isdir(fn)

def dirname(fn):
	return os.path.dirname(fn)

def rootname(fn):
	return os.path.splitext(fn)[0]

def rootnamewopath(fn):
	return os.path.splitext(filenamewopath(fn))[0]

def filepath(fn):
	return os.path.split(fn)[0]

def filenamewopath(fn):
	return os.path.split(fn)[1]

def replaceext(fn,ext):
	return '%s%s' % (rootname(fn),ext)

def extname(fn):
	return os.path.splitext(fn)[1]


def addtail2rootname(name,tail):
	root, ext = os.path.splitext(name)
	return root + tail + ext

def join(a,*p):
	return os.path.join(a,*p)

def mkdir(dn,logger):
	try:
		logger.info('creating a directory "%s"...' % dn)
		os.mkdir(dn)
	except OSError as oserr:
		assert False, oserr

def rmdirsforcefully(dn,logger):
	if not exists(dn):
		logger.info("No directory to remove: %s" % dn2str(dn))
		return

	assert isdir(dn), '"%s" must be a directory.' % dn

	try:
		logger.info("removing `%s' and its subdirectories..." % dn2str(dn))
		for root, dirs, files in os.walk(dn,topdown=False):
			for name in files:
				os.remove(join(root, name))
			os.rmdir(root)
	except Exception as e:
		logger.info('Exception raised while removing %s...' % dn2str(dn))
		assert False, e

def remove_files_except(dirname,*re_filenames):
	for root, dirs, files in os.walk(dirname,topdown=False):
		for name in files:
			filename = join(root,name)
			if isfile(filename) and not reg_match(name,*re_filenames):
				os.remove(filename)

def draw_text(axis,msg):
	axis.text(0.0,0.0,msg,ha='center',va='center')

def reg_match(s,*regexps):
	return any([re.match(regexp,s) for regexp in regexps])

def linterp( xlist, xp, yp ):
	xp[0], yp[0]
	assert len(xp) > 1 and len(yp) > 1, (xp,yp)

	xl, yl = unique_sortlists(xp,yp)

	assert len(xl) > 1 and len(yl) > 1, ('FATAL!', xl, yl )

	res = []
	interp = []
	for x in xlist:
		i = False
		if x < xl[0]:
			y = int_ext_div(x,xl[:2],yl[:2])
		elif x > xl[-1]:
			y = int_ext_div(x,xl[-2:],yl[-2:])
		else:
			y = np.interp(x,xl,yl)
			i = True
		res.append(y)
		interp.append(i)

	return res, interp

def int_ext_div(x,xl,yl):
	assert len(xl) == 2 and len(yl) == 2, (xl,yl)
	assert xl[0] != xl[1], xl

	alpha = (float(x)-xl[0])/(xl[1]-xl[0])
	beta = 1.-alpha

	return beta*yl[0] + alpha*yl[1]

def int_ext_division( x1, x2, r ):
	return (1.-r) * x1 + float( r ) * x2


def istype(x,ts):
	try:
		ts[0]
	except TypeError:
		ts = [ts]

	for t in ts:
		if isinstance(x,t):
			return True

	return False

def fn2str(x):
	return '"%s"' % x

def files_w_ext(dirname,ext):
	assert isdir(dirname), '%s must be a directory name.' % dn2str( dirname )

	res = []
	for name in os.listdir(dirname):
		filename = os.path.join(dirname,name)
		if isfile(filename) and extname(filename) == ext:
			res.append(filename)

	return res

def files_startswith( dirname, s ):
	assert isdir(dirname), '%s must be a directory name.' % dn2str( dirname )

	res = []
	for name in os.listdir(dirname):
		filename = os.path.join(dirname,name)
		if isfile(filename) and name.startswith( s ):
			res.append(filename)

	return res


def dirs_startswith( dirname, s ):
	assert isdir( dirname ), '%s must be a directory name.' % dn2str( dirname )

	res = []
	for name in os.listdir( dirname ):
		filename = os.path.join( dirname, name )
		if isdir( filename ) and name.startswith( s ):
			res.append( filename )

	return res


def isvec(x):
	return type(x) is tuple or type(x) is list

def list2str(l,delim=', '):
	return delim.join([str(x) for x in l])

def isstrlist(l,n=None):
	return islist(l,(str,),n)

def isnumlist(l,n=None):
	return islist(l,(float,int),n)

def islist(l,tps,n=None):
	if not type(l) is list:
		return False

	if not n is None and len(l) != n:
		return False

	for e in l:
		if not istype(e,tps):
			return False

	return True

def is_range(xmin,xmax):
	return xmin < xmax

def intersect_range(xmin1,xmax1,xmin2,xmax2):
	assert is_range(xmin1,xmax1) and is_range(xmin2,xmax2), (xmin1,xmax1,xmin2,xmax2)

	xmin, xmax = max(xmin1,xmin2), min(xmax1,xmax2)
	if is_range(xmin,xmax):
		return xmin, xmax
	else:
		return None

def is_similar_range(x1,x2,y1,y2,l=.2):
	assert x1 <= x2 and y1 <= y2, (x1,x2,y1,y2)
	return y1 < int_div(x1,x2,l)\
		and x1 < int_div(y1,y2,l)\
		and int_div(x2,x1,l) < y2\
		and int_div(y2,y1,l) < x2

def int_div(x,y,l):
	return (1.0-l)*x + float(l)*y

def l2floatl(l):
	return [float(x) for x in l]

def isunique(l):
	return len(set(l)) == len(l)

def iswithinrange(x,r):
	return x >= min(r) and x <= max(r)

#tic()
#import numpy as np
#toc( 'importing numpy', __file__ )

def sortlists(l1,l2):
	l1 = np.array(l1)
	l2 = np.array(l2)

	idx = l1.argsort()

	return list(l1[idx]), list(l2[idx])

def unique_sortlists(xs,ys):
	assert len(xs) == len(ys), (len(xs),len(ys))

	m = {}
	for i, x in enumerate(xs):
		if not m.has_key(x):
			m[x] = ys[i]

	keys = m.keys()
	keys.sort()
	xlist, ylist = [], []

	for key in keys:
		xlist.append(key)
		ylist.append(m[key])

	return xlist, ylist

def to_tuple(t):
	try:
		t[0]
		return tuple(t)
	except TypeError:
		return (t,)

	assert False, 'FATAL!'


def _check_field_reqs(data,field_reqs,errmsg):

	can_reqs = {}

	class A:
		pass

	def field_req_format_check():
		for field in field_reqs:
			req = field_reqs[field]
			assert len(req) == 2, errmsg
			typs, fcns = req[0], req[1]
			if typs:
				try:
					typs[0]
				except TypeError:
					typs = (typs,)

			if fcns:
				try:
					fcns[0]
				except TypeError:
					fcns = (fcns,)

			for typ in typs:
				assert type(typ) is type or type(typ) is type(A), (errmsg,typ,type(typ),typs)

			can_reqs[field] = (typs,fcns)

	def _check_field(field):
		val = data[field]
		typs, fcns = can_reqs[field]

		assert istype(val,typs), (errmsg,val,data,typs,fcns)
		for fcn in fcns:
			assert fcn(val), (errmsg,val,fcn,fcns)

	## main body

	field_req_format_check()

	for field in field_reqs:
		if data.has_key(field):
			_check_field(field)

def _dict_field_check(d1,d2,errmsg=''):
	assert type(d1) is dict and type(d2) is dict, (d1,d2)
	if len(d2) == 0:
		return

	if not errmsg:
		errmsg = 'cutil._dict_field_check error'

	try:
		nopts = 0
		for key in d2:
			chck = d2[key]
			if len(chck) == 3 and chck[2] == 'optional':
				nopts += 1
				if not d1.has_key(key):
					continue

			val = d1[key]
			typs = chck[0]
			try:
				typs[0]
			except TypeError as e:
				typs = (typs,)

			typepass = False
			typenames = []
			for tp in typs:
				typenames.append(str(tp))
				if isinstance(val,tp):
					typepass = True
					break

			assert typepass, "%s:`%s' must be either of %s." % (errmsg,key,' or '.join(typenames))

			if not chck[1]:
				continue

			chkfcns = chck[1]
			try:
				chkfcns[0]
			except TypeError:
				chkfcns = (chkfcns,)

			for fcn in chkfcns:
				fcn(val)

		s1 = set(d1.keys())
		s2 = set(d2.keys())
		assert s1.issubset(s2), ('%s: there are extra fields.' % errmsg, s1.difference(s2),s1,s2)

	except KeyError as ke:
		dbmsg(218, errmsg, ke,d1)
		raise

idfcn = lambda x : x
ngfcn = lambda x : -x

def getjobstatuses(jobids):
	assert type(jobids) is list, 'FATAL!'
	assert len(jobids) > 0, 'FATAL!'

	res = {}
	for status in Outfile.statusnames():
		res[status] = []

	for jobid in jobids:
		status = Outfile('%s.out' % jobid).status()
		res[status].append(jobid)

	return res


def alldoneorexited(jobids,name,fakesim):
	tnum = len(jobids)
	assert tnum > 0, (tnum,jobids)

	if fakesim:
		statuses = getjobstatuses(jobids)
		numdone, numexit = len(statuses['done']), len(statuses['exited'])
		#print '%s: %d/%d job(s) done or exited:' % (name,dnum,tnum), disp_nonemtpy_fields_only(statuses)
	else:
		jobstats, jobstat_cat = JjobStatus.getstatuses(jobids)
		numdone, numexit = len(jobstat_cat['DONE']), len(jobstat_cat['EXIT'])

	dnum = numdone + numexit
	assert dnum <= tnum, (dnum,tnum,jobids,name)
	return dnum == tnum, numdone, numexit

def disp_nonemtpy_fields_only(d):
	l = []
	keys = d.keys()
	keys.sort()
	for key in keys:
		val = d[key]
		if val:
			l.append('%s: %s' % (key,val))

	return list2str(l)


def get_fig_axes( nrow, ncol\
		, l_margin, r_margin, b_margin, t_margin\
		, sp_width, sp_height\
		, wpadding, hpadding, **kargs ):
	tic()
	import matplotlib.pyplot as pl
	toc( 'importing matplotliab.pyplot', __file__ )


	width = l_margin + r_margin + sp_width * ncol + wpadding * ( ncol - 1 )
	height = b_margin + t_margin + sp_height * nrow + hpadding * ( nrow - 1 )

	fig = pl.figure( figsize = ( width, height ) )
	#fig = pl.figure( figsize = ( width, height ), facecolor = 'lightgoldenrodyellow' )
	#fig = pl.figure( figsize = ( width, height ), facecolor = 'red' )
	#print 'fig.get_facecolor() =', fig.get_facecolor()

	for i in range( nrow ):
		for j in range( ncol ):
			l = ( l_margin + ( sp_width + wpadding ) * j ) / width
			b = ( b_margin + ( sp_height + hpadding ) * ( nrow - i - 1 ) ) / height
			w = sp_width / width
			h = sp_height / height

			fig.add_axes( [ l, b, w, h ], **kargs )

	return fig

def change_ticklabel_size( ax, fontsize ):
	for ticklabel in ax.get_majorticklabels():
		ticklabel.set_fontsize( fontsize )
		ticklabel.set_color('orange')
		print(dir(ticklabel))
		assert False

def set_majorticklables( axis, **kargs ):
	for ticklabel in axis.get_majorticklabels():
		ticklabel.set( **kargs )

def set_major_formatter( axis, format ):
	tic()
	import matplotlib.ticker as ticker
	toc( 'importing matplotlib.ticker', __file__ )

	axis.set_major_formatter( ticker.FormatStrFormatter( format ) )

class Expression:
	Uop = { '-':ngfcn, '()':idfcn, '""':idfcn, "''":idfcn }
	Bop = { '+' : lambda x,y : x+y\
		, '-': lambda x,y : x-y\
		, '*': lambda x,y : x*y\
		, '/': lambda x,y : x/y\
		}

	def __init__(self,*pargs):
		self.na = len(pargs)
		self.ps = pargs
		assert self.na >= 1 and self.na <= 3, pargs

	def value(self,m={}):
		p = self.ps
		if self.na == 1:
			val = p[0]
			if type(val) is str:
				if not m.has_key(val):
					assert False, ('KeyError',val,m)
				return float(m[val])
			else:
				return float(val)
		elif self.na == 2:
			return Expression.Uop[p[0]](p[1].value(m))
		elif self.na == 3:
			return Expression.Bop[p[0]](p[1].value(m),p[2].value(m))
		else:
			assert False, 'FATAL!'


	def __repr__(self):
		if self.na == 1:
			try:
				return '%g' % self.ps[0]
			except:
				return self.ps[0]

		if self.na == 2:
			if self.ps[0] == '-':
				return '-' + str(self.ps[1])
			elif self.ps[0] == '()':
				return '(' + str(self.ps[1]) + ')'
			elif self.ps[0] == "''":
				return "'" + str(self.ps[1]) + "'"
			elif self.ps[0] == '""':
				return '"' + str(self.ps[1]) + '"'
			else:
				assert False, self.ps

		if self.na == 3:
			return str(self.ps[1]) + self.ps[0] + str(self.ps[2])

class MyDict(dict):
	def __setitem__(self,f,v):
		try:
			dict.__getitem__(self,f)
		except KeyError:
			dict.__setitem__(self,f,v) ##
			return

		print('cutil.py:677: (f,v) =', (f,v))
		assert False, ('field has already been assigned.',f,v)

class DefDict( dict ):
	def __init__( self, *p, **k ):
		self.default_value = k.pop( 'default', None )
		dict.__init__( self, *p, **k )

	def __getitem__( self, f ):
		try:
			return dict.__getitem__( self, f )
		except KeyError:
			self[f] = copy.deepcopy( self.default_value )
			return self[f]
		


class JjobStatus:

	StatNames = ['not found', 'PEND', 'RUN', 'DONE', 'EXIT']

	@staticmethod
	def statusnames():
		return JjobStatus.StatNames

	@staticmethod
	def getstatuses(jobids):
		assert isstrlist(jobids)

		jobstats = dict.fromkeys(jobids,None)
		jobstat_cat = {}
		for stat in JjobStatus.StatNames:
			jobstat_cat[stat] = []

		cnt = 0
		while True:
			#for line in popen('jjobs %s' % ' '.join(jobids)):
			for line in popen('bjobs %s' % ' '.join(jobids)):
				tokens = line.split()
				assert len(tokens) > 0, (tokens,line)
				if tokens[0] in jobids:
					assert len(tokens) >= 3, (tokens,line)
					jobid, stat = tokens[0], tokens[2]

				elif tokens[0] == 'Job':
					assert len(tokens) == 5, tokens
					assert tokens[2] == 'is' and tokens[3] == 'not' and tokens[4] == 'found', tokens
					t = tokens[1]
					assert len(t) >= 3 and t[0] == '<' and t[-1] == '>', t
					jobid = t[1:-1]
					stat = 'not found'

				else:
					continue

				assert jobstats[jobid] is None, (jobid,jobstats)
				assert stat in JjobStatus.StatNames, stat
				jobstats[jobid] = stat
				jobstat_cat[stat].append(jobid)

			if all([not jobstats[jobid] is None for jobid in jobstats]):
				break

			cnt += 1
			if cnt > 10:
				assert False, ( 'bjobs cannot get the information of all the job even after trying 10 times', jobstats )

		for jobid in jobstats:
			assert not jobstats[jobid] is None, (jobid,jobstats)

		assert sum([len(jobstat_cat[x]) for x in jobstat_cat]) == len(jobids), (jobstat_cat,jobids)

		return jobstats, jobstat_cat


class Outfile:
	@staticmethod
	def statusnames():
		return ['no file', 'done', 'unknown', 'exited']

	def __init__(self,filename):
		self.filename = filename

		assert len(self.filename) > 4 and self.filename[-4:] == '.out'

	def status(self):
		assert not os.path.exists(self.filename) or os.path.isfile(self.filename), "`%s' must be a file" % self.filename

		if not os.path.exists(self.filename):
			return 'no file'

		##JEKAI
		fid = open(self.filename)
		for line in fid:
			line = line.strip()
			if line == 'Successfully completed.':
				return 'done'

			if line.startswith('>info:') and line.endswith('hspice job aborted'):
				return 'exited'

			if line.startswith('Exited'):
				return 'exited'
		fid.close()
		return 'unknown'

class BaseClass:
	def __init__(self,mustfields={},fieldreq={}):
		self.checkdatafields(mustfields)
		self.check_field_reqs(fieldreq)

	def checkdatafields(self,mustfields,errmsg=''):
		_dict_field_check(self.__dict__, mustfields, errmsg)

	def check_field_reqs(self,fieldreq,errmsg=''):
		_check_field_reqs(self.__dict__, fieldreq, errmsg)

class DefException(Exception):
	def __init__(self,*pargs,**kargs):
		Exception.__init__(self,*pargs,**kargs)

if __name__ == "__main__":
	tlist = ['expression']
	tlist = ['sort list']
	tlist = ['getjobstatuses']
	tlist = ['range']
	tlist = ['rms']
	tlist = ['rm_except']
	tlist = ['last_mono_start_idx']
	tlist = ['same_text_files']

	if 'same_text_files' in tlist:
		print(same_text_files('mpopt.py','mpopt.py'))
		print(same_text_files('mpopt.py','fmf.py'))
		print(same_text_files('mpopt.py','mpopt1.py'))

	if 'last_mono_start_idx' in tlist:

		ll = []
		ll.append([2,1,2,3])
		ll.append([1,1,2,3])
		ll.append([0,1,2,3])
		ll.append([3,2,1,0])
		ll.append([3,2,1,2])
		ll.append([3,2,1,2,3,4,5])
		ll.append([3,2,1,2,3,4,5,4])
		for l in ll:
			s = last_mono_start_idx(l)
			print('list, idx, new_list =', (l,s,l[s:]))

	if 'rm_except' in tlist:
		remove_files_except('/user/ciropt/sim_file_examples'\
				, '.*\.tr[0-9]+$'\
				, '.*\.mt[0-9]+$' )


	if 'rms' in tlist:
		print('rms([1,2,3]) =', rms([1,2,3]))


	if 'range' in tlist:
		print(is_range(1,2), is_range(1,1), is_range(2,1))
		print(intersect_range(1,2,1.5,3))
		print(intersect_range(1,3,3,4))
		print(intersect_range(3,4,1,3))


	if 'getjobstatuses' in tlist:
		directory = os.curdir
		filelist = [ x for x in os.listdir(directory) if x.endswith('.out')]
		jobnames = [ x[:-4] for x in filelist]

		print(getjobstatuses(jobnames))

	if 'sort list' in tlist:
		print(linterp([1.5,4,2.5,.5],[1,2,3],[5,2,3]))
		print(linterp([1.5,4,2.5,.5],[2,3,1],[2,3,5]))


	if 'expression' in tlist:
		ex = Expression
		l = []
		l.append(ex(1))
		l.append(ex('a'))
		l.append(ex('-',ex(2.)))
		l.append(ex('()',ex(2.)))
		l.append(ex('""',ex(2.)))
		l.append(ex("''",ex(2.)))
		l.append(ex('+',ex(2.),ex(3)))
		l.append(ex('-',ex(2.),ex(3)))
		l.append(ex('*',ex(2.),ex(3)))
		l.append(ex('/',ex(2.),ex(3)))

		m = {'a':3}

		print([x.value(m) for x in l ])

		print(isnumlist([1,2,3]))
		print(isnumlist([1,2,'r']))
		print(isnumlist([1,2,3],3))
		print(isnumlist([1,2,3],2))
		print(isstrlist(['sd']))
		print(isstrlist(['sd','sdf']))
		print(isstrlist(['sd','sdf',123]))
		print(islist([Expression('s')],Expression))
		print(islist([Expression('s'),2],Expression))
		print(islist([Expression('s'),Expression(1)],(Expression,)))
		print(islist([Expression('s'),Expression(1)],(Expression,int,float)))
		print(islist([Expression('s'),Expression(1)],(Expression,int,float),3))

class EmptyClass:
	pass

