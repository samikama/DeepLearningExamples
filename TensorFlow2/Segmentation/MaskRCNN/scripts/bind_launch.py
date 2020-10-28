import sys
import subprocess
import os
import socket
from argparse import ArgumentParser, REMAINDER, ArgumentTypeError


def parse_args():
  """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
  parser = ArgumentParser(description="PyTorch distributed training launch "
                          "helper utilty that will spawn up "
                          "multiple distributed processes")

  # Optional arguments for the launch helper
  parser.add_argument("--nnodes",
                      type=int,
                      default=1,
                      help="The number of nodes to use for distributed "
                      "training")
  parser.add_argument("--node_rank",
                      type=int,
                      default=0,
                      help="The rank of the node for multi-node distributed "
                      "training")
  parser.add_argument("--nproc_per_node",
                      type=int,
                      default=1,
                      help="The number of processes to launch on each node, "
                      "for GPU training, this is recommended to be set "
                      "to the number of GPUs in your system so that "
                      "each process can be bound to a single GPU.")
  parser.add_argument("--master_addr",
                      default="127.0.0.1",
                      type=str,
                      help="Master node (rank 0)'s address, should be either "
                      "the IP address or the hostname of node 0, for "
                      "single node multi-proc training, the "
                      "--master_addr can simply be 127.0.0.1")
  parser.add_argument("--master_port",
                      default=29500,
                      type=int,
                      help="Master node (rank 0)'s free port that needs to "
                      "be used for communciation during distributed "
                      "training")
  parser.add_argument('--no_hyperthreads',
                      action='store_true',
                      help='Flag to disable binding to hyperthreads')
  parser.add_argument('--no_membind',
                      action='store_true',
                      help='Flag to disable memory binding')

  # non-optional arguments for binding
  parser.add_argument("--nsockets_per_node",
                      type=int,
                      required=True,
                      help="Number of CPU sockets on a node")
  parser.add_argument("--ncores_per_socket",
                      type=int,
                      required=True,
                      help="Number of CPU cores per socket")

  def strAsBool(s):
    if isinstance(s, bool):
      return s
    if s.lower() in ("yes", "true", "1", "y", "t"):
      return True
    elif s.lower() in ("no", "false", "0", "n", "f"):
      return False
    else:
      raise ArgumentTypeError("Need boolean type")

  parser.add_argument('--direct_launch',
                      nargs='?',
                      const=True,
                      default=False,
                      type=strAsBool,
                      help='Flag to disable numactl launch')

  # positional
  parser.add_argument("training_script",
                      type=str,
                      help="The full path to the single GPU training "
                      "program/script to be launched in parallel, "
                      "followed by all the arguments for the "
                      "training script")

  # rest from the training program
  parser.add_argument('training_script_args', nargs=REMAINDER)
  return parser.parse_args()


def main():
  #from mpi4py import MPI
  args = parse_args()

  # variables for numactrl binding
  NSOCKETS = args.nsockets_per_node
  #NGPUS_PER_SOCKET = max(args.nproc_per_node // args.nsockets_per_node,1)
  NGPUS_PER_SOCKET = args.nproc_per_node // args.nsockets_per_node
  NCORES_PER_GPU = args.ncores_per_socket // NGPUS_PER_SOCKET

  # world size in terms of number of processes
  dist_world_size = args.nproc_per_node * args.nnodes

  # set PyTorch distributed related environmental variables
  current_env = os.environ.copy()
  # for (i,v) in sorted(current_env.items(),key=lambda x:x[0]):
  #     print(f"{i} : {v}")

  processes = []
  curr_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))

  def startRank(local_rank):
    #for local_rank in range(curr_rank, args.nproc_per_node):
    # each process's rank
    dist_rank = args.nproc_per_node * args.node_rank + local_rank
    # form numactrl binding command
    cpu_ranges = [
        local_rank * NCORES_PER_GPU, (local_rank + 1) * NCORES_PER_GPU - 1,
        local_rank * NCORES_PER_GPU +
        (NCORES_PER_GPU * NGPUS_PER_SOCKET * NSOCKETS),
        (local_rank + 1) * NCORES_PER_GPU +
        (NCORES_PER_GPU * NGPUS_PER_SOCKET * NSOCKETS) - 1
    ]

    numactlargs = []
    if args.no_hyperthreads:
      numactlargs += ["--physcpubind={}-{}".format(*cpu_ranges[0:2])]
    else:
      numactlargs += ["--physcpubind={}-{},{}-{}".format(*cpu_ranges)]

    if not args.no_membind:
      memnode = local_rank // NGPUS_PER_SOCKET
      numactlargs += ["--membind={}".format(memnode)]

    # spawn the processes
    cmd = [ "/usr/bin/numactl" ] \
        + numactlargs \
        + [ sys.executable,
            "-u",
            args.training_script
          ] \
        + args.training_script_args
    print("SAMI Launching with numactl", " ".join(cmd))
    process = subprocess.Popen(cmd, env=current_env)
    processes.append(process)

  print("Direct launch is ",args.direct_launch)
  if not args.direct_launch:
    startRank(curr_rank)
  else:
    cmd = [sys.executable, "-u", args.training_script
          ] + args.training_script_args
    print("SAMI Launching directly", " ".join(cmd))
    process = subprocess.Popen(cmd, env=current_env)
    processes.append(process)
  for process in processes:
    process.wait()


if __name__ == "__main__":
  main()
