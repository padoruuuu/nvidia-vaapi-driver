# Maintainer: padoruuuu

pkgname='libva-nvidia-driver-git'
pkgver=0.0.12.r0.g0000000
pkgrel=1
pkgdesc='VA-API implementation that uses NVDEC as a backend (git version, fork by padoruuuu)'
arch=('x86_64')
url='https://github.com/padoruuuu/nvidia-vaapi-driver'
license=('MIT')
depends=('gst-plugins-bad-libs' 'libegl')
makedepends=('git' 'meson' 'ffnvcodec-headers' 'libva')
provides=('libva-nvidia-driver')
conflicts=('libva-nvidia-driver' 'libva-vdpau-driver')
source=("${pkgname}::git+https://github.com/padoruuuu/nvidia-vaapi-driver.git")
sha256sums=('SKIP')

pkgver() {
    cd ${pkgname}
    printf "0.0.12.r%s.g%s\n" "$(git rev-list --count HEAD)" "$(git rev-parse --short HEAD)"
}

build() {
    cd ${pkgname}
    arch-meson . build
    meson compile -C build
}

package() {
    cd ${pkgname}
    meson install -C build --destdir "${pkgdir}"
    install -Dm644 COPYING "${pkgdir}"/usr/share/licenses/"${pkgname}"/LICENSE
}
