add_rules("mode.debug", "mode.release")

target("cpp_mnist")
    set_kind("binary")
    add_files("src/*.cpp")
