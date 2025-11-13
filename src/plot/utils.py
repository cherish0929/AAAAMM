import os
import matplotlib.pyplot as plt

def save_fig(filename):
    """
    安全保存图片（增强）：如果参数 filename 是目录，则提示用户重新输入文件名。
    """
    # 先检查用户输入是否是目录
    if os.path.isdir(filename):
        print(f"❌ 输入的路径 '{filename}' 是一个目录，而不是文件名！")
        folder = filename.rstrip("/")
        while True:
            new_name = input("请输入文件名（如 figure.png）: ").strip()

            if new_name == "":
                print("❌ 文件名不能为空，请重新输入。")
                continue

            if os.path.sep in new_name:
                print("❌ 请不要输入路径，只输入文件名，例如 'a.png'。")
                continue

            if "." not in new_name:
                new_name += ".png"

            filename = os.path.join(folder, new_name)

            if os.path.isdir(filename):
                print("❌ 输入的是目录名，请重新输入文件名。")
                continue

            print(f"🔁 保存路径更新为：{filename}")
            break

    orig_filename = filename
    folder = os.path.dirname(filename)

    while os.path.exists(filename):
        print(f"⚠️ 文件 '{filename}' 已存在！")
        print("请选择操作：")
        print("   [y] 覆盖保存")
        print("   [n] 取消保存")
        print("   [r] 重新输入文件名保存（只需输入文件名，无需路径）")
        choice = input("请输入(y/n/r): ").strip().lower()

        if choice == "y":
            break
        elif choice == "n":
            print("🛑 已取消保存。")
            return
        elif choice == "r":
            while True:
                new_name = input("请输入新的文件名（如 new.png）: ").strip()

                if new_name == "":
                    print("❌ 文件名不能为空，请重试。")
                    continue

                if os.path.sep in new_name or new_name.endswith("/"):
                    print("❌ 请不要输入路径，只输入文件名。")
                    continue

                if "." not in new_name:
                    new_name += ".png"

                new_filename = os.path.join(folder, new_name)
                if os.path.isdir(new_filename):
                    print("❌ 输入的是目录名，请重新输入。")
                    continue
                filename = new_filename
                print(f"🔁 重命名为：{filename}")
                break
        else:
            print("❌ 无效输入，请重新选择。")

    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"✅ 保存成功！图片已存于：{filename}")

