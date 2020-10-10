# Import dependices.
from ecmwfapi import ECMWFDataServer
import click

@click.command()
@click.option('--key', prompt='Key', help="ECMWF API Key")
@click.option('--email', prompt='Email', help="Email Address")
def main(key, email):
    # Define server.
    server = ECMWFDataServer(
        url="https://api.ecmwf.int/v1", key=key, email=email
    )

    # Retrieve data.
    server.retrieve({
        "class": "ti",
        "dataset": "tigge",
        "date": "2019-01-01/to/2019-01-31",
        "expver": "prod",
        "grid": "1.0/1.0",
        "levelist": "200/250/300/500/700/850/925/1000",
        "levtype": "pl",
        "origin": "ecmf",
        "param": "130/131/132/133/156",
        "step": "0/6/12/18/24/30/36/42/48/54/60/66/72/78/84/90/96/102/108/114/120",
        "time": "00:00:00",
        "type": "fc",
        "target": "output",
    })

if __name__ == '__main__':
    main()
