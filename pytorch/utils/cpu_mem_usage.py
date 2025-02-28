import time
_FIELDS = ['VmRSS', 'VmHWM', 'VmSize', 'VmPeak']


def get_memory(str1):
	'''
	returns the current and peak, real and virtual memories
	used by the calling linux python process, in Bytes
	'''

	# read in process info
	with open('/proc/self/status', 'r') as file:
		lines = file.read().split('\n')

	# container of memory values (_FIELDS)
	values = {}

	# check all process info fields
	for line in lines:
		if ':' in line:
			name, val = line.split(':')

			# collect relevant memory fields
			if name in _FIELDS:
				values[name] = int(val.strip().split(' ')[0])  # strip off "kB"
				values[name] /= 1024  # convert to MB

	# check we collected all info
	assert len(values)==len(_FIELDS)
	print(str1+str(values))
	return str1+str(values)

def get_time_memory(tic,str1):
	'''
	returns the current and peak, real and virtual memories
	used by the calling linux python process, in Bytes
	'''

	# read in process info
	with open('/proc/self/status', 'r') as file:
		lines = file.read().split('\n')

	# container of memory values (_FIELDS)
	values = {}

	# check all process info fields
	for line in lines:
		if ':' in line:
			name, val = line.split(':')

			# collect relevant memory fields
			if name in _FIELDS:
				values[name] = int(val.strip().split(' ')[0])  # strip off "kB"
				values[name] /= 1024  # convert to MB

	# check we collected all info
	assert len(values)==len(_FIELDS)
	toc = time.time()
	print(str1 + ' step Time(s): {:.4f}'.format(toc - tic))
	return str1+str(values)


if __name__=='__main__':
	# a simple test
	print(get_memory(""))
	mylist = [1.5] * 2 ** 30
	print(get_memory(""))
	del mylist
	print(get_memory(""))

