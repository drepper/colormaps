import moreland

maps = {
	'smooth-diverging': moreland.make_smooth_diverging,
	'bent-diverging': moreland.make_bent_diverging,
	'viridis': moreland.make_viridis,
	'plasma': moreland.make_plasma,
	'black-body': moreland.make_black_body,
	'inferno': moreland.make_inferno,
	'kindlmann': moreland.make_kindlmann,
	'extended-kindlmann': moreland.make_extended_kindlmann
}

if __name__ == '__main__':
	import sys
	try:
		if len(sys.argv) < 3:
			raise Exception('too few arguments')
		if not sys.argv[1] in maps:
			raise Exception('unknown map')
		length = int(sys.argv[2])

		byte = False
		real = False
		for sel in sys.argv[3:]:
			if sel == 'byte':
				byte = True
			elif sel == 'float':
				real = True
			else:
				raise Exception('unknown selector "{}"'.format(sel))
		if not byte and not real:
			byte = True
			real = True

		maps[sys.argv[1]]().write_table(length = length, byte = byte, real = real)
	except Exception as e:
		print('ERROR:', str(e))
		print('Usage: {} MAP LENGTH [byte|float]...'.format(sys.argv[0]))
		print('Known MAPs:', ', '.join([s for s in maps]))
		sys.exit(1)
