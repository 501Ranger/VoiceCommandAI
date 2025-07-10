from setuptools import find_packages, setup

package_name = 'llm_ros_interface'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/' + package_name, ['package.xml']),
        #('share/' + package_name + '/resource', ['data/' + package_name]),
        # 如果有其他非python文件需要安装，例如配置文件
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name', # 填写您的名字
    maintainer_email='your.email@example.com', # 填写您的邮箱
    description='ROS2 interface for LLM integration with audio ASR',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'llm_subscriber = llm_ros_interface.llm_subscriber:main', # 这行很重要
        ],
    },
)