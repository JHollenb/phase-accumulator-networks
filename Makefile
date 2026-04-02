
src.py:
	@find src -name "*.py" -exec bash -c 'echo "#{}"; cat {}' \; >> $@
