from __future__ import print_function,division

def main():

    import topaz.commands.boxes_to_coordinates 
    module_list = [
                    topaz.commands.boxes_to_coordinates 
                  ]

    import argparse
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(title='commands', metavar='<command>')
    subparsers.required = 'True'
    subparsers.dest = 'command'
    for module in module_list:
        this_parser = subparsers.add_parser(module.name, help=module.help)
        module.add_arguments(this_parser)
        this_parser.set_defaults(func=module.main)


    args = parser.parse_args()

    args.func(args)


if __name__ == '__main__':
    main()


