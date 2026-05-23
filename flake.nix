{
  pkgs ? import <nixpkgs> { },
}:

pkgs.mkShell {
  buildInputs = [
    (pkgs.python3.withPackages (
      ps: with ps; [
      ]
    ))
  ];

  shellHook = ''
    export QT_QPA_PLATFORM=wayland
  '';
}
