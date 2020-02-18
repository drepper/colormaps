maps = bent-cool-warm black-body extended-kindlmann inferno kindlmann plasma smooth-cool-warm viridis

all:

clean:
	-rm -f ${addsuffix -table-byte-????.csv,${maps}} ${addsuffix -table-float-????.csv,${maps}}
	-rm -f ${addsuffix .svg,${maps}}
	-rm -fr __pycache__

.PHONY: all clean
