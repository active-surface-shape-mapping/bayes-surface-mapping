{
  description = "Develop Python on Nix with uv";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-25.11";
  };

  outputs =
    { nixpkgs, ... }:
    let
      inherit (nixpkgs) lib;
      forAllSystems = lib.genAttrs lib.systems.flakeExposed;
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          fhs = pkgs.buildFHSEnv {
              name = "uv-fhs-env";
              targetPkgs = pkgs: with pkgs; [ pkgs.uv libz ];
              runScript = "./splash.sh";
            };
        in
        {
          default = fhs.env;
        }
      );
    };
}
