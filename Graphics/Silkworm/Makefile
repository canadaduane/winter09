GHC_OPTIONS=--make -framework Accelerate

# Compile and link the game executable
Build:
	ghc -o SilkwormGame $(GHC_OPTIONS) Main.hs

# Make the SilkwormGame.app application for Mac OS X
MacSilkwormGame:
	ghc -o SilkwormGame $(GHC_OPTIONS) Main.hs
	utils/macosx-app SilkwormGame
	cp -r Resources/ SilkwormGame.app/Contents/Resources/

# Run the Mac OS X app
run: MacSilkwormGame
	open SilkwormGame.app


# Make the Hipmunk Playground application for Mac OS X
MacPlayground: Playground
	utils/macosx-app Playground

Playground: Playground.hs
	ghc $(GHC_OPTIONS) Playground.hs

RunPlayground: MacPlayground
	open Playground.app


# Cleanup
clean:
	rm -f SilkwormGame Playground
	rm -f *.hi *.o
	rm -f Silkworm/*.hi Silkworm/*.o
	rm -rf Playground.app SilkwormGame.app