OBJS=Silkworm/Constants.o Silkworm/Init.o Silkworm/Misc.o

# Make the Silkworm application for Mac OS X
MacSilkworm: SilkwormGame
	utils/macosx-app SilkwormGame

SilkwormGame: Main.hs $(OBJS)
	ghc -o SilkwormGame --make Main.hs $(OBJS)

Run: MacSilkworm
	open SilkwormGame.app

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
	rm -f SilkwormGame Playground
	rm -f *.hi *.o
	rm -f Silkworm/*.hi Silkworm/*.o
	rm -rf Playground.app Silkworm.app