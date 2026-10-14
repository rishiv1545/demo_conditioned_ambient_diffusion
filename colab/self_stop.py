"""Stop (unassign) this Colab VM from inside it, so billing ends even when no watcher on the Mac is running.

Runs on the VM with the colab CLI installed in its own venv (/content/colabcli; the training env is untouched), after
colab/job.py has finished and flushed Drive:

    /content/colabcli/bin/python colab/self_stop.py <endpoint>

`dcad.py run` sets this up: it installs the venv, uploads the CLI's OAuth token (~/.config/colab-cli/token.json, kept
on the VM's local disk only, never on Drive; it disappears with the VM) and passes the session's endpoint to job.py
--self_stop. The unassign call is the one `colab stop` makes.
"""
import sys

from colab_cli.common import state


def main():
    endpoint = sys.argv[1]
    state.client.unassign(endpoint)
    print("unassigned", endpoint, flush=True)


if __name__ == "__main__":
    main()
