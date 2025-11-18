#!/usr/bin/env python3
"""
Local 2FA Authenticator
Secure, encrypted local storage for TOTP codes
"""

import argparse
import base64
import getpass
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import pyotp
import qrcode
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class Local2FA:
    """Encrypted local 2FA authenticator."""

    def __init__(self, storage_file: str = "~/.local_2fa_vault.enc"):
        self.storage_file = Path(storage_file).expanduser()
        self.storage_file.parent.mkdir(parents=True, exist_ok=True)
        self.accounts: Dict[str, str] = {}
        self.fernet: Optional[Fernet] = None

    def _derive_key(self, password: str, salt: bytes) -> bytes:
        """Derive encryption key from password."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))

    def _encrypt_data(self, data: str, password: str) -> bytes:
        """Encrypt data with password."""
        salt = os.urandom(16)
        key = self._derive_key(password, salt)
        fernet = Fernet(key)
        encrypted = fernet.encrypt(data.encode())
        return salt + encrypted

    def _decrypt_data(self, encrypted_data: bytes, password: str) -> str:
        """Decrypt data with password."""
        salt = encrypted_data[:16]
        encrypted = encrypted_data[16:]
        key = self._derive_key(password, salt)
        fernet = Fernet(key)
        return fernet.decrypt(encrypted).decode()

    def load(self, password: str) -> bool:
        """Load and decrypt stored accounts."""
        if not self.storage_file.exists():
            self.accounts = {}
            return True

        try:
            encrypted_data = self.storage_file.read_bytes()
            decrypted = self._decrypt_data(encrypted_data, password)
            self.accounts = json.loads(decrypted)
            return True
        except Exception as e:
            print(f"Error loading vault: {e}")
            return False

    def save(self, password: str) -> bool:
        """Encrypt and save accounts."""
        try:
            data = json.dumps(self.accounts, indent=2)
            encrypted = self._encrypt_data(data, password)
            self.storage_file.write_bytes(encrypted)
            self.storage_file.chmod(0o600)  # Owner read/write only
            return True
        except Exception as e:
            print(f"Error saving vault: {e}")
            return False

    def add_account(self, name: str, secret: str):
        """Add a new 2FA account."""
        # Validate secret
        try:
            pyotp.TOTP(secret).now()
        except Exception:
            raise ValueError("Invalid secret key")

        self.accounts[name] = secret
        print(f"Added account: {name}")

    def remove_account(self, name: str):
        """Remove a 2FA account."""
        if name in self.accounts:
            del self.accounts[name]
            print(f"Removed account: {name}")
        else:
            print(f"Account not found: {name}")

    def list_accounts(self):
        """List all accounts."""
        if not self.accounts:
            print("No accounts configured")
            return

        print("\nConfigured accounts:")
        for i, name in enumerate(sorted(self.accounts.keys()), 1):
            print(f"  {i}. {name}")
        print()

    def get_code(self, name: str) -> Optional[str]:
        """Get current TOTP code for account."""
        if name not in self.accounts:
            print(f"Account not found: {name}")
            return None

        secret = self.accounts[name]
        totp = pyotp.TOTP(secret)
        code = totp.now()

        # Calculate time remaining
        time_remaining = 30 - (int(time.time()) % 30)

        return code, time_remaining

    def show_qr_code(self, name: str, issuer: str = "Local2FA"):
        """Generate QR code for account setup."""
        if name not in self.accounts:
            print(f"Account not found: {name}")
            return

        secret = self.accounts[name]
        totp = pyotp.TOTP(secret)
        uri = totp.provisioning_uri(name=name, issuer_name=issuer)

        # Generate QR code
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(uri)
        qr.make(fit=True)

        # Print to terminal
        qr.print_ascii(invert=True)
        print(f"\nManual entry key: {secret}")
        print(f"URI: {uri}")

    def generate_new_secret(self) -> str:
        """Generate a new random secret."""
        return pyotp.random_base32()

    def watch_codes(self, interval: int = 1):
        """Watch mode - continuously display all codes."""
        print("Press Ctrl+C to exit\n")

        try:
            while True:
                os.system('clear')
                print("═" * 60)
                print("  LOCAL 2FA AUTHENTICATOR - WATCH MODE")
                print("═" * 60)
                print()

                current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                time_remaining = 30 - (int(time.time()) % 30)

                print(f"Time: {current_time}")
                print(f"Refresh in: {time_remaining} seconds")
                print()

                if not self.accounts:
                    print("No accounts configured")
                else:
                    for name in sorted(self.accounts.keys()):
                        code, remaining = self.get_code(name)
                        bar_length = 20
                        filled = int((remaining / 30) * bar_length)
                        bar = "█" * filled + "░" * (bar_length - filled)

                        print(f"{name:30s} {code}  [{bar}] {remaining}s")

                print()
                print("─" * 60)
                print("Press Ctrl+C to exit")

                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n\nExiting watch mode...")


def main():
    parser = argparse.ArgumentParser(
        description="Local 2FA Authenticator - Secure TOTP code generator"
    )
    parser.add_argument(
        "--vault",
        default="~/.local_2fa_vault.enc",
        help="Path to encrypted vault file"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Add account
    add_parser = subparsers.add_parser("add", help="Add new 2FA account")
    add_parser.add_argument("name", help="Account name")
    add_parser.add_argument("--secret", help="Secret key (will be generated if not provided)")
    add_parser.add_argument("--qr", action="store_true", help="Show QR code after adding")

    # Remove account
    remove_parser = subparsers.add_parser("remove", help="Remove 2FA account")
    remove_parser.add_argument("name", help="Account name")

    # List accounts
    subparsers.add_parser("list", help="List all accounts")

    # Get code
    get_parser = subparsers.add_parser("get", help="Get TOTP code for account")
    get_parser.add_argument("name", help="Account name")

    # Show QR code
    qr_parser = subparsers.add_parser("qr", help="Show QR code for account")
    qr_parser.add_argument("name", help="Account name")
    qr_parser.add_argument("--issuer", default="Local2FA", help="Issuer name")

    # Watch mode
    watch_parser = subparsers.add_parser("watch", help="Watch mode - display all codes")
    watch_parser.add_argument("--interval", type=int, default=1, help="Update interval in seconds")

    # Generate secret
    subparsers.add_parser("generate", help="Generate a new random secret")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize 2FA manager
    tfa = Local2FA(args.vault)

    # Generate secret doesn't need vault
    if args.command == "generate":
        secret = tfa.generate_new_secret()
        print(f"Generated secret: {secret}")
        print(f"\nUse this with: {sys.argv[0]} add <name> --secret {secret}")
        return

    # Get password
    if tfa.storage_file.exists():
        password = getpass.getpass("Vault password: ")
        if not tfa.load(password):
            print("Failed to unlock vault. Wrong password?")
            sys.exit(1)
    else:
        print("Creating new vault...")
        password = getpass.getpass("Choose vault password: ")
        password_confirm = getpass.getpass("Confirm password: ")
        if password != password_confirm:
            print("Passwords don't match!")
            sys.exit(1)
        tfa.accounts = {}

    # Execute command
    try:
        if args.command == "add":
            secret = args.secret if args.secret else tfa.generate_new_secret()
            tfa.add_account(args.name, secret)
            tfa.save(password)

            if args.qr or not args.secret:
                print("\nQR code for easy mobile app setup:")
                tfa.show_qr_code(args.name)

        elif args.command == "remove":
            tfa.remove_account(args.name)
            tfa.save(password)

        elif args.command == "list":
            tfa.list_accounts()

        elif args.command == "get":
            result = tfa.get_code(args.name)
            if result:
                code, remaining = result
                print(f"\n{args.name}: {code}")
                print(f"Valid for {remaining} more seconds\n")

        elif args.command == "qr":
            tfa.show_qr_code(args.name, args.issuer)

        elif args.command == "watch":
            tfa.watch_codes(args.interval)

    except KeyboardInterrupt:
        print("\n\nCancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
