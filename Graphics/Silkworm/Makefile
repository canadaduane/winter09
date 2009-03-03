OBJS=Silkworm.o

# Make the Silkworm application for Mac OS X
MacSilkworm: Silkworm
	utils/macosx-app Silkworm

Silkworm: Main.hs $(OBJS)
	ghc -o Silkworm --make Main.hs $(OBJS)

Init.o: Init.hs
	ghc -c Init.hs

Misc.o: Misc.hs
	ghc -c Misc.hs

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