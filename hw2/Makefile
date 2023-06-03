all:

pretty: black prettier

black:
	isort --profile black $(CURDIR)
	black $(CURDIR)

prettier:
	prettier --write $(CURDIR)
