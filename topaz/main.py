from __future__ import print_function,division

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)

    import topaz.commands.train
    import topaz.commands.segment
    import topaz.commands.extract
    import topaz.commands.boxes_to_coordinates 

    module_groups = [('Particle picking',
                      [topaz.commands.train,
                       topaz.commands.segment,
                       topaz.commands.extract,
                      ]
                     ),
                     ('Conversion utilities',
                      [topaz.commands.boxes_to_coordinates,
                      ]
                     ),
                    ]

    description = []
    for group,module_list in module_groups:
        description.append(group + ':')
        for module in module_list:
            description.append('  ' + module.name + '\t' + module.help)
        description.append('')
    description = '\n'.join(description)
    
    subparsers = parser.add_subparsers(title='commands', metavar='<command>'
                                      , description=description)
    subparsers.required = 'True'
    subparsers.dest = 'command'
    for group,module_list in module_groups:
        for module in module_list:
            this_parser = subparsers.add_parser(module.name) #, help=module.help)
            module.add_arguments(this_parser)
            this_parser.set_defaults(func=module.main)


    #subparsers = parser.add_subparsers(title='Conversion utilities', metavar='<command>')
    #subparsers.required = 'True'
    #subparsers.dest = 'command'


    args = parser.parse_args()

    args.func(args)


if __name__ == '__main__':
    main()


