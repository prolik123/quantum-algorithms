import unittest
import importlib.metadata
import os
import re
from packaging.version import parse


class TestEnvironmentSetup(unittest.TestCase):
    def setUp(self):
        # Dynamically resolve the path to requirements.txt relative to this file's location.
        # This ensures the test works regardless of where you run it from.
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.requirements_path = os.path.join(base_dir, "requirements.txt")
        self.required_versions = self._parse_requirements()

    def _parse_requirements(self) -> dict:
        """
        Parses requirements directly from the requirements.txt file.
        Ignores empty lines and comments.
        """
        if not os.path.exists(self.requirements_path):
            self.fail(f"❌ File not found: {self.requirements_path}")

        reqs = {}
        with open(self.requirements_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Ignore comments and empty lines
                if not line or line.startswith('#'):
                    continue

                # Use regex to separate the package name from the operator and version
                # Captures: package_name >=, ==, <=, ~= version
                match = re.match(r'^([a-zA-Z0-9\-_]+)\s*[>=<~]+\s*(.*)$', line)
                if match:
                    package_name = match.group(1).strip()
                    version = match.group(2).strip()
                    reqs[package_name] = version
                else:
                    # If a package is listed without a specific version (e.g., "qiskit")
                    reqs[line] = "0.0.0"

        return reqs

    def test_packages_installed_and_versions_correct(self):
        """
        Checks if the required packages are installed and meet the version constraints.
        """
        # Ensure the requirements dictionary is not empty
        self.assertTrue(
            len(self.required_versions) > 0,
            "❌ The requirements.txt file appears to be empty or poorly formatted."
        )

        for package, min_version in self.required_versions.items():
            with self.subTest(package=package):
                try:
                    # Retrieve the installed package version
                    installed_version = importlib.metadata.version(package)

                    if min_version != "0.0.0":
                        self.assertGreaterEqual(
                            parse(installed_version),
                            parse(min_version),
                            f"❌ Package '{package}' is too old: {installed_version} (required >= {min_version})"
                        )
                        print(f"✅ {package.ljust(15)}: OK (required {min_version}, installed {installed_version})")
                    else:
                        print(f"✅ {package.ljust(15)}: OK (installed {installed_version}, no minimum requirement)")

                except importlib.metadata.PackageNotFoundError:
                    # pip sometimes stores packages with underscores instead of hyphens internally
                    alt_package = package.replace('-', '_')
                    try:
                        installed_version = importlib.metadata.version(alt_package)
                        print(f"✅ {package.ljust(15)}: OK (found as '{alt_package}' - {installed_version})")
                    except importlib.metadata.PackageNotFoundError:
                        self.fail(f"❌ Package '{package}' is not installed!")


if __name__ == '__main__':
    unittest.main(verbosity=2)