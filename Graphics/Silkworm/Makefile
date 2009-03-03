OBJS=Silkworm.o

# Make the Silkworm application for Mac OS X
MacSilkworm: Silkworm
	utils/macosx-app Silkworm

Silkworm: Main.hs $(OBJS)
	ghc --make Main.hs $(OBJS)

Silkworm.o: Silkworm.hs
	ghc -c Silkworm.hs

Run: MacSilkworm
	open Silkworm.app

run: Run


# Make the Hipmunk Playground application for Mac OS X
MacPlayground: Playground
	utils/macosx-app Playground

Playground: Playground.hs
	ghc --make Playground.hs

RunPlayground: MacPlayground
	open Playground.app


# Cleanup
clean:
	rm -f Main Silkworm Playground *.hi *.o
	rm -rf Playground.app Silkworm.app