{
  description = "Gait analysis prototype: data pipeline, model, notebooks and Gradio demo";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      # nixpkgs marks gradio broken when mcp >= 2; only gradio's optional MCP-server mode is
      # affected, which the demo does not use. Allow exactly these two, and skip gradio's tests
      # (a broken package is not in the binary cache, so it is built locally), and its pillow<13
      # pin (nixpkgs ships pillow 13).
      pkgsFor = system: import nixpkgs {
        inherit system;
        config.problems.handlers = {
          gradio.broken = "ignore";
          gradio-sans-reverse-dependencies.broken = "ignore";
        };
        overlays = [
          (final: prev: {
            pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
              (pyfinal: pyprev: {
                gradio = pyprev.gradio.overrideAttrs { doInstallCheck = false; dontCheckRuntimeDeps = true; };
              })
            ];
          })
        ];
      };
      forAll = f: nixpkgs.lib.genAttrs systems (system: f (pkgsFor system));

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
        # video -> pose -> inference demo
        opencv4
        onnxruntime
        gradio
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
            pkgs.ffmpeg # demo: browser-playable (H.264) annotated video
          ];
          shellHook = ''
            export JUPYTER_PATH=${kernelFor pkgs}''${JUPYTER_PATH:+:$JUPYTER_PATH}
            export PYTHONPATH=$PWD/src''${PYTHONPATH:+:$PYTHONPATH}
          '';
        };
      });
    };
}
