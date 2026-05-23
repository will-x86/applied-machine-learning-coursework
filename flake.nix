{
  description = "flake";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ ];
        };
      in
      {
        devShell = pkgs.mkShell {
          buildInputs = with pkgs; [
            #xorg.libxcb
          ];
          nativeBuildInputs = with pkgs; [
          ];

          #            export DISPLAY=:0
          #            export WAYLAND_DISPLAY=wayland-1
          #            export SDL_VIDEODRIVER=x11
          shellHook = ''
                      export LD_LIBRARY_PATH="${
                        pkgs.lib.makeLibraryPath [
                          pkgs.libxcb
                          pkgs.libGL
                          #pkgs.SDL2
                          pkgs.glib
                          #pkgs.libGLU

                        ]
                      }:$LD_LIBRARY_PATH"

            export HSA_OVERRIDE_GFX_VERSION=11.0.0
            export QT_QPA_PLATFORM=xcb
            export GPU_BACKEND=rocm

          '';
        };
      }
    );
}
