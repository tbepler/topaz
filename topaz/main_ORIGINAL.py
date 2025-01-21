from __future__ import absolute_import, print_function, division

def generate_description(module_groups, linewidth=78, indent='  ', delim='  '):

    """
    description = []
    for group,module_list in module_groups:
        description.append(group + ':')
        for module in module_list:
            description.append('  ' + module.name + '\t' + module.help)
        description.append('')
    description = '\n'.join(description)
    """

    description = []

    names = []
    for group,module_list in module_groups:
        for module in module_list:
            names.append(module.name)

    ## name column width
    name_width = max(len(name) for name in names)
    desc_width = linewidth - len(indent) - name_width - len(delim)
    
    for group,module_list in module_groups:
        description.append(group + ':')
        for module in module_list:
            name = module.name
            descriptor = module.help
            ## first line includes name, pad to name_width
            name = name + ' '*(name_width-len(name))
            ## take tokens from descriptor up to desc_width to generate lines
            width = 0
            line_tokens = []
            for token in descriptor.split():
                if width + len(token) > desc_width:
                    if width > 0: # push current tokens to line
                        line = indent + name + delim + ' '.join(line_tokens)
                        description.append(line)
                        name = ' '*name_width
                    width = 0
                    line_tokens = []
                line_tokens.append(token)
                width += len(token)
            if width > 0:
                line = indent + name + delim + ' '.join(line_tokens)
                description.append(line)
        description.append('')
    return '\n'.join(description)


def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, fromfile_prefix_chars='@')

    import topaz
    parser.add_argument('--version', action='version', version=topaz.__version__)

    import topaz.commands.train
    import topaz.commands.segment
    import topaz.commands.extract
    import topaz.commands.precision_recall_curve

    import topaz.commands.downsample
    import topaz.commands.normalize
    import topaz.commands.preprocess
    import topaz.commands.denoise
    import topaz.commands.denoise3d

    import topaz.commands.gui

    import topaz.commands.convert
    import topaz.commands.split
    import topaz.commands.particle_stack
    import topaz.commands.train_test_split

    # deprecated
    import topaz.commands.scale_coordinates
    import topaz.commands.boxes_to_coordinates 
    import topaz.commands.star_to_coordinates 
    import topaz.commands.coordinates_to_star
    import topaz.commands.coordinates_to_boxes
    import topaz.commands.coordinates_to_eman2_json
    import topaz.commands.star_particles_threshold

    module_groups = [('Particle picking',
                      [topaz.commands.train,
                       topaz.commands.segment,
                       topaz.commands.extract,
                       topaz.commands.precision_recall_curve,
                      ]
                     ),
                     ('Image processing',
                      [topaz.commands.downsample,
                       topaz.commands.normalize,
                       topaz.commands.preprocess,
                       topaz.commands.denoise,
                       topaz.commands.denoise3d,
                      ]
                     ),
                     ('File utilities',
                      [
                       topaz.commands.convert,
                       topaz.commands.split,
                       topaz.commands.particle_stack,
                       topaz.commands.train_test_split,
                      ]
                     ),
                     ('GUI',
                      [
                       topaz.commands.gui,
                      ]
                     ),
                     ('[Deprecated]',
                      [
                       topaz.commands.scale_coordinates,
                       topaz.commands.boxes_to_coordinates,
                       topaz.commands.star_to_coordinates,
                       topaz.commands.coordinates_to_star,
                       topaz.commands.coordinates_to_boxes,
                       topaz.commands.coordinates_to_eman2_json,
                       topaz.commands.star_particles_threshold,
                      ]
                     ),
                    ]

    description = generate_description(module_groups)
    
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


