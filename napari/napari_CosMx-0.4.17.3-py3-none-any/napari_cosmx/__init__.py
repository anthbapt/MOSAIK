# NanoString Technologies, Inc.
# Software License Agreement for Non-Commercial Use
# By downloading, installing, accessing, modifying or otherwise making use of the Program (defined below), you agree to be bound by the terms and conditions of this Software License Agreement for Non-Commercial Use (this “License”).
# 1.	DEFINITIONS
# 1.1.	“Affiliate” means, with respect to an individual or entity, another individual or entity: (i) on whose behalf such individual or entity is acting, or (ii) that exercises control, is controlled by, or is under common control with such individual or entity. For the purposes of this definition, the term “control” means the right, whether by ownership, exercise of voting rights, contract, or otherwise, to direct the actions of an individual or entity.
# 1.2.	“Distribute” means to distribute, share, make available, or otherwise provide the Program or Modified Program, as applicable, or access thereto (including via a computer network) to any third party.
# 1.3.	“Licensor” means the individual or entity licensing the rights granted in this License.
# 1.4.	“Licensee” or “you” means the individual or entity receiving or exercising the rights granted under this License, provided that the individual or entity is not a NanoString Competitor.
# 1.5.	“Non-Commercial Use” means any use where profit or other commercial benefit is not a direct or indirect motive or intended result.
# 1.6.	“Modified Program” means a derivative work of, or a work that is based on, uses or incorporates, the Program (whether or not in combination with other works, materials or content).
# 1.7.	“NanoString” means NanoString Technologies, Inc.
# 1.8.	“NanoString Competitor” means any individual or entity that directly or indirectly competes with NanoString or any of NanoString’s Affiliates or whose Affiliate directly or indirectly competes with NanoString or any of NanoString’s Affiliates.
# 1.9.	“Program” means the copyrightable work of authorship, program, code, or software licensed under this License.
# 2.	LICENSE 
# 2.1.	Grant. Subject to the terms and conditions of this License, Licensor hereby grants to Licensee a worldwide, royalty-free, non-exclusive, revocable license to: (a) use, Distribute, and reproduce the Program, and (b) use, create, Distribute, and reproduce Modified Programs, in each case, solely for your internal, Non-Commercial Use. No rights are granted to NanoString Competitors.
# 2.2.	No Endorsement. Nothing in this License may be construed as permission to assert or imply that Licensor, NanoString, or other contributors to the Program sponsors, endorses, or is otherwise connected with the Licensee or the entity or institution that Licensee represents.
# 2.3.	Trademarks. Trademark rights are not licensed to you under this License.
# 2.4.	Grant of Patent License. Subject to the terms and conditions of this License, NanoString hereby grants to you a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable (except as stated in this section) patent license to make, have made, use, import, and otherwise transfer the Program, where such license applies only to those patent claims licensable by NanoString that are necessarily infringed by Licensee alone or by combination of its modification(s) to the Program or Modified Program to which such modification(s) was submitted. If you institute patent litigation against any entity (including a cross-claim or counterclaim in a lawsuit) alleging that the Program, Modified Program, or a modification incorporated within the Program or a Modified Program constitutes direct or contributory patent infringement, then any patent licenses granted to you under this License for the Program or any such Modified Program shall terminate as of the date such litigation is filed.
# 3.	CONDITIONS TO THE RIGHT TO DISTRIBUTE
# 3.1.	Notices. If you Distribute the Program or a Modified Program in any form, you must also provide to the recipient:
# 3.1.1.	a copy of this License; and 
# 3.1.2.	for Modified Programs, prominent notices identifying the portions of the Modified Program that have been modified, stating that you have modified the Program.
# 3.2.	Attribution. Except as otherwise expressly permitted under this License, you must keep intact, and you may not modify or remove, any notices, disclaimers, or attributions included in or provided with the Program. In addition, you must also include a prominent hypertext link back to NanoString’s website at www.nanostring.com. 
# 3.3.	License. You may only Distribute the Program or the Modified Program under the terms of this License (or any later version, at your election). You may not offer or impose any additional or different terms or conditions that, or take any measures to, restrict the exercise of the rights granted under this License.
# 4.	NO REPRESENTATIONS OR WARRANTIES; LIMITATIONS OF LIABILITY
# 4.1.	Disclaimer. UNLESS OTHERWISE AGREED BY LICENSOR IN WRITING, TO THE FULLEST EXTENT PERMITTED BY APPLICABLE LAW, LICENSOR OFFERS THE PROGRAM AS-IS AND MAKES NO REPRESENTATIONS OR WARRANTIES OF ANY KIND WITH REGARD TO THE PROGRAM, WHETHER EXPRESS, IMPLIED, STATUTORY OR OTHERWISE, INCLUDING WITHOUT LIMITATION, WARRANTIES OF TITLE, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT. THE LICENSOR DOES NOT REPRESENT OR WARRANT THAT THE PROGRAM WILL BE ERROR FREE AND DOES NOT PROMISE THAT ANY SUCH ERRORS WILL BE CORRECTED.
# SOME JURISDICTIONS DO NOT ALLOW FOR THE EXCLUSION OF IMPLIED WARRANTIES, SO THE FOREGOING MAY NOT APPLY TO YOU.
# 4.2.	Limitation of Liability. TO THE FULLEST EXTENT PERMITTED BY APPLICABLE LAW, IN NO EVENT WILL THE LICENSOR OR NANOSTRING BE LIABLE TO YOU UNDER ANY LEGAL THEORY FOR ANY DAMAGES OF ANY KIND, INCLUDING ANY SPECIAL, INCIDENTAL, CONSEQUENTIAL, PUNITIVE OR EXEMPLARY DAMAGES ARISING OUT OF OR RELATED TO THE PROGRAM OR USE THEREOF, EVEN IF LICENSOR OR NANOSTRING HAS BEEN ADVISED OF THE POSSIBILITY OR LIKELIHOOD OF SUCH DAMAGES.
# 5.	MISCELLANEOUS
# 5.1.	Right to Enforce. NanoString is an express third-party beneficiary of this License and will be entitled to enforce the provisions of this License as if it were a party hereto. 
# 5.2.	Waiver; Amendment. No term or provision hereof will be considered waived by the Licensor, and no breach excused by Licensor, unless such waiver or consent is in writing and signed by an authorized representative of Licensor.  The waiver by Licensor of, or consent by Licensor to, a breach of any provision of this License by the Licensee, will not constitute, operate or be construed as a waiver of, consent to, or excuse of any other or subsequent breach by Licensee.  This License may be amended or modified only by an agreement in writing signed by an authorized representative of each of Licensor and Licensee.

BETA_UM_PER_PX = 0.1203
BETA_MM_PER_PX = BETA_UM_PER_PX/1000
BETA_PX_PER_MM = 1/BETA_MM_PER_PX

ALPHA_UM_PER_PX = 0.1681
ALPHA_MM_PER_PX = ALPHA_UM_PER_PX/1000
ALPHA_PX_PER_MM = 1/ALPHA_MM_PER_PX

DASH_UM_PER_PX = 0.18
DASH_MM_PER_PX = DASH_UM_PER_PX/1000
DASH_PX_PER_MM = 1/DASH_MM_PER_PX

DEFAULT_COLORMAPS = {
    'DAPI': 'blue',
    'DNA': 'blue',
    'PanCK': 'green',
    'U': 'blue',
    'G': 'green',
    None: 'gray'
}

OTHER_KEYS = ['labels', 'protein', 'composite', 'targets', 'fovgrid']

from ._reader import napari_get_reader

from ._function import napari_experimental_provide_function
