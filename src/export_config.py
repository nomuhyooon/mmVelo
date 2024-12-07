import subprocess

def export_conda_packages_to_setup_cfg(output_file="setup.cfg"):
    # `conda list` コマンドを実行してパッケージ情報を取得
    result = subprocess.run(["conda", "list", "--export"], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error retrieving package list:", result.stderr)
        return

    # パッケージ情報をパースして setup.cfg フォーマットに変換
    lines = result.stdout.splitlines()
    packages = []
    for line in lines:
        if not line.startswith("#") and line.strip():  # コメント行を無視
            package_info = line.split("=")
            if len(package_info) >= 2:  # パッケージ名とバージョンが存在する場合
                package_name = package_info[0]
                version = package_info[1]
                packages.append(f"{package_name}=={version}")

    # setup.cfg ファイルとして書き込み
    with open(output_file, "w") as f:
        f.write("[options]\n")
        f.write("install_requires =\n")
        for package in packages:
            f.write(f"    {package}\n")

    print(f"setup.cfg has been successfully created at {output_file}.")

# 実行
export_conda_packages_to_setup_cfg()
