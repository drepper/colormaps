import moreland


def plot_n_write(map):
	map.plot()
	map.write_tables()


plot_n_write(moreland.make_smooth_diverging())
plot_n_write(moreland.make_bent_diverging())
plot_n_write(moreland.make_viridis())
plot_n_write(moreland.make_plasma())
plot_n_write(moreland.make_black_body())
plot_n_write(moreland.make_inferno())
plot_n_write(moreland.make_kindlmann())
plot_n_write(moreland.make_extended_kindlmann())
