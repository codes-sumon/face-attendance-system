"""
Generate a self-signed SSL certificate for development.
Uses the cryptography library to create cert.pem and key.pem files.
"""
import os
import socket
from datetime import datetime, timedelta, timezone

from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend


def generate_self_signed_cert(
    cert_file="cert.pem",
    key_file="key.pem",
    days_valid=365,
    alt_names=None,
):
    """
    Generate a self-signed certificate using the cryptography library.
    """
    # Generate a private key
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend(),
    )

    # Subject and issuer (self-signed, so same)
    cn = alt_names[0] if alt_names else "localhost"
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "State"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, "City"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Face Attendance System Dev"),
        x509.NameAttribute(NameOID.COMMON_NAME, cn),
    ])

    # Build SAN list
    san_entries = [x509.DNSName(cn)]
    for name in (alt_names or []):
        if name != cn:
            san_entries.append(x509.DNSName(name))

    now = datetime.now(timezone.utc)

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(1000)
        .not_valid_before(now)
        .not_valid_after(now + timedelta(days=days_valid))
        .add_extension(
            x509.SubjectAlternativeName(san_entries),
            critical=False,
        )
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None),
            critical=True,
        )
        .sign(private_key, hashes.SHA256(), backend=default_backend())
    )

    # Write private key
    with open(key_file, "wb") as f:
        f.write(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

    # Write certificate
    with open(cert_file, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    print(f"✅ SSL certificate generated: {cert_file}")
    print(f"✅ SSL private key generated: {key_file}")
    print(f"   Valid for: {days_valid} days")
    print(f"   Common Name: {cn}")


if __name__ == "__main__":
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    alt_names = ["localhost", hostname, local_ip]
    print(f"🔐 Generating self-signed SSL certificate...")
    print(f"   Hostname: {hostname}")
    print(f"   Local IP: {local_ip}")
    generate_self_signed_cert(alt_names=alt_names)
