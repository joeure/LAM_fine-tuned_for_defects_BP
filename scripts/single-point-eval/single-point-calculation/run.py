import logging, lammps, time
import argparse

def calculate(input_script, log_file, maceOrDPA3:bool):
    # device = "cuda" if torch.backends.cuda
    # logging.info(f"Using device: {device}") # No this kind of cmds in Torch 1.13.1
    logging.info(f"Using LAMMPS input script: {input_script}")
    logging.info(f"Using LAMMPS log file: {log_file}")

    if maceOrDPA3:
        cmd = [
            "-log", log_file,
            "-k", "on", "g", "1",       # turn on Kokkos with 1 GPU
            "-sf", "kk",                # activate Kokkos-accelerated styles
        ]
    else:
        cmd = [
            "-log", log_file,   # keep a log
            # (no "-k on g 1", no "-sf kk")
        ]

    tries, delay = 5, 15
    for attempt in range(1, tries + 1):
        logging.critical("▶️ attempt %d/%d", attempt, tries)
        try:
            lmp = lammps.lammps(cmdargs=cmd)
            lmp.command("info styles minimize")
            lmp.file(input_script)  # run the input script
            logging.info("LAMMPS run completed successfully.")
            lmp.close()
            lmp.finalize()  # close LAMMPS instance
            return 0                                 # success
        except RuntimeError as err:               # LAMMPS threw an error
            if "cudaMalloc" in str(err):          # OOM in Kokkos/CUDA
                logging.warning("⚠️  OOM, sleep %ds …", delay)
            else:
                logging.error("❌  %s retry after %ds", err, delay)
            time.sleep(delay)
    else:
        raise RuntimeError("gave up after %d tries" % tries)

def main(model:str, log_basename:str):
    logs_dir = f"prepare_space/{model}/logs"
    calculate(f"prepare_space/{model}/in.singlepoint.lmp", 
              f"{logs_dir}/{(log_basename or f'sp_{model}.log')}", 
              maceOrDPA3=("mace" in model.lower()))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run the Single Point Lammps Running")
    ap.add_argument("--model", required=True, help="model name")
    ap.add_argument("--log_basename", required=True, help="base name for the lammps log file")
    args = ap.parse_args()
    
    main(args.model, args.log_basename)
    