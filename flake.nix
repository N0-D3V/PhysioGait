{
  description = "Gait analysis prototype: data pipeline + notebooks";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAll = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});

      pythonFor = pkgs: pkgs.python3.withPackages (ps: with ps; [
        # pipeline
        numpy
        scipy
        pandas
        pyarrow
        openpyxl # Toronto Participants.xlsx
        tqdm
        # model
        torch
        # notebooks
        matplotlib
        nbformat
        nbconvert
        ipykernel
        ipywidgets
        jupyterlab
      ]);

      # "Python (gait)" kernel, the kernel name the notebooks are saved with
      kernelFor = pkgs: pkgs.runCommand "gait-jupyter-kernel" { } ''
        mkdir -p $out/kernels/gait
        cat > $out/kernels/gait/kernel.json <<JSON
        {
          "argv": ["${pythonFor pkgs}/bin/python", "-Xfrozen_modules=off", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
          "display_name": "Python (gait)",
          "language": "python",
          "metadata": { "debugger": true }
        }
        JSON
      '';
    in
    {
      packages = forAll (pkgs: {
        default = pythonFor pkgs;
        python = pythonFor pkgs;
        jupyter-kernel = kernelFor pkgs;
      });

      devShells = forAll (pkgs: {
        default = pkgs.mkShell {
          packages = [
            (pythonFor pkgs)
            pkgs._7zz # Ferber ric_data.zip is Deflate64, which unzip can't read
            pkgs.unzip
            pkgs.curl
          ];
          shellHook = ''
            export JUPYTER_PATH=${kernelFor pkgs}''${JUPYTER_PATH:+:$JUPYTER_PATH}
            export PYTHONPATH=$PWD/src''${PYTHONPATH:+:$PYTHONPATH}
          '';
        };
      });
    };
}
