#!/usr/bin/env bash
# Install the fluxvla_infer environment for deploy/ (pistar06 ROS2 real-robot
# inference). Delegates the ML dependencies to scripts/install_env.sh
# real-only instead of duplicating them, then adds what that script doesn't
# cover: deploy/'s pinned visualization packages, and ROS2 Jazzy itself
# (real-only's own ROS check is for ROS1 Noetic/rospy, a different stack).
#
# ROS2_INSTALL controls the Jazzy install step: auto (default) installs it
# only if rclpy isn't already importable, always forces a reinstall attempt,
# never only checks and warns. Installing ROS2 Jazzy runs `apt-get` as root
# (via sudo) and adds the ros2.org apt source system-wide.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="${PYTHON}"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
else
  PYTHON_BIN="python"
fi

ROS_DISTRO_NAME="jazzy"
ROS_SETUP="${ROS_SETUP:-/opt/ros/${ROS_DISTRO_NAME}/setup.bash}"
ROS2_INSTALL="${ROS2_INSTALL:-auto}"

case "${ROS2_INSTALL}" in
  auto|always|never) ;;
  *)
    echo "Error: ROS2_INSTALL must be one of: auto, always, never" >&2
    exit 1
    ;;
esac

as_root() {
  if [[ "${EUID}" == "0" ]]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    echo "Error: need root or sudo to install ROS2 ${ROS_DISTRO_NAME} system packages." >&2
    return 1
  fi
}

rclpy_importable() {
  if [[ -f "${ROS_SETUP}" ]]; then
    set +u
    # shellcheck disable=SC1090
    source "${ROS_SETUP}"
    set -u
  fi
  "${PYTHON_BIN}" -c "import rclpy" >/dev/null 2>&1
}

install_ros2_jazzy() {
  if [[ "$(uname -s)" != "Linux" ]]; then
    echo "Error: ROS2 ${ROS_DISTRO_NAME} auto-install only supports Linux (Ubuntu 24.04 Noble)." >&2
    return 1
  fi
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "Error: apt-get not found; install ROS2 ${ROS_DISTRO_NAME} manually:" >&2
    echo "       https://docs.ros.org/en/${ROS_DISTRO_NAME}/Installation.html" >&2
    return 1
  fi

  echo "Installing ROS2 ${ROS_DISTRO_NAME} (ros-${ROS_DISTRO_NAME}-ros-base) via apt."
  as_root apt-get update
  as_root apt-get install -y software-properties-common curl
  as_root add-apt-repository -y universe
  as_root apt-get update

  local version_codename ros_apt_source_version deb_url deb_path
  version_codename="$(. /etc/os-release && echo "${VERSION_CODENAME}")"
  ros_apt_source_version="$(
    curl -fsSL https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
      | grep -F '"tag_name"' | head -n1 | awk -F'"' '{print $4}'
  )"
  if [[ -z "${ros_apt_source_version}" ]]; then
    echo "Error: could not determine the latest ros-apt-source release version." >&2
    return 1
  fi

  deb_url="https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ros_apt_source_version}/ros2-apt-source_${ros_apt_source_version}.${version_codename}_all.deb"
  deb_path="/tmp/ros2-apt-source.deb"
  curl -fsSL -o "${deb_path}" "${deb_url}"
  verify_deb_checksum "${deb_url}" "${deb_path}"
  as_root apt-get install -y "${deb_path}"
  as_root apt-get update
  as_root apt-get install -y "ros-${ROS_DISTRO_NAME}-ros-base" python3-rosdep
}

# Verify the downloaded .deb against the sha256sum GitHub publishes alongside
# release assets (<asset>.sha256, a convention ros-apt-source's release
# workflow follows). Refuses to install an unverified package unless the
# caller explicitly opts in via ROS2_INSTALL_ALLOW_UNVERIFIED=1 -- installing
# ROS2_INSTALL's .deb runs apt-get as root, so a tampered/MITM'd download
# must not be silently trusted.
verify_deb_checksum() {
  local deb_url="$1" deb_path="$2"
  local checksum_url="${deb_url}.sha256"
  local checksum_path="${deb_path}.sha256"

  if ! curl -fsSL -o "${checksum_path}" "${checksum_url}" 2>/dev/null; then
    if [[ "${ROS2_INSTALL_ALLOW_UNVERIFIED:-0}" == "1" ]]; then
      echo "Warning: no ${checksum_url} published; installing ${deb_path} UNVERIFIED" \
        "(ROS2_INSTALL_ALLOW_UNVERIFIED=1)." >&2
      return 0
    fi
    echo "Error: could not fetch a checksum for ${deb_path} from ${checksum_url};" >&2
    echo "       refusing to apt-get install an unverified package downloaded over" >&2
    echo "       the network. Re-run with ROS2_INSTALL_ALLOW_UNVERIFIED=1 to bypass" >&2
    echo "       (not recommended), or install ROS2 ${ROS_DISTRO_NAME} manually." >&2
    return 1
  fi

  local expected actual
  expected="$(awk '{print $1}' "${checksum_path}")"
  actual="$(sha256sum "${deb_path}" | awk '{print $1}')"
  if [[ -z "${expected}" || "${expected}" != "${actual}" ]]; then
    echo "Error: checksum mismatch for ${deb_path}" >&2
    echo "       expected: ${expected:-<empty>}" >&2
    echo "       actual:   ${actual}" >&2
    echo "       refusing to install a package that does not match its published checksum." >&2
    return 1
  fi
  echo "    checksum verified: ${deb_path} matches ${checksum_url}"
}

echo "==> [1/3] Training / real-robot inference dependencies (real-only)"
echo "    (the 'rospy' warning this may print is for ROS1 and does not apply to deploy/'s ROS2 stack)"
bash "$REPO_ROOT/scripts/install_env.sh" real-only "$@"

echo "==> [2/3] deploy/ visualization dependencies"
"$PYTHON_BIN" -m pip install -r "$SCRIPT_DIR/requirements-visualization.txt"

echo "==> [3/3] ROS2 ${ROS_DISTRO_NAME}"
if [[ "${ROS2_INSTALL}" == "never" ]]; then
  if rclpy_importable; then
    echo "    rclpy is importable."
  else
    echo "Warning: rclpy is not importable and ROS2_INSTALL=never; skipping install." >&2
    echo "         Install ROS2 ${ROS_DISTRO_NAME} manually and source ${ROS_SETUP}." >&2
  fi
elif [[ "${ROS2_INSTALL}" == "always" ]] || ! rclpy_importable; then
  install_ros2_jazzy
  if rclpy_importable; then
    echo "    rclpy is importable after install."
  else
    echo "Warning: rclpy is still not importable after installing ROS2 ${ROS_DISTRO_NAME}." >&2
    echo "         Check that ${ROS_SETUP} exists and matches ${PYTHON_BIN}'s Python version." >&2
  fi
else
  echo "    rclpy is already importable; skipping install (ROS2_INSTALL=auto)."
fi

echo "==> deploy/ environment ready. Edit deploy/launch.sh, then run: ./deploy/launch.sh"
