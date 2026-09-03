Import("env")

openocd = env.PioPlatform().get_package_dir("tool-openocd") + "/bin/openocd"
scripts = env.PioPlatform().get_package_dir("tool-openocd") + "/openocd/scripts"

env.Replace(
    UPLOADCMD=(
        f'"{openocd}" '
        f'-s "{scripts}" '
        '-f interface/stlink.cfg '
        '-f target/stm32f4x.cfg '
        '-c "program {$SOURCE} verify reset; shutdown;"'
    )
)